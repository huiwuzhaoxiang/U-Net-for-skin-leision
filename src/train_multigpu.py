import torch
import torchvision
from PIL import Image
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model import *

# 初始化分布式环境
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

# ------------------------- 数据预处理修正 -------------------------
trans_totensor = torchvision.transforms.Compose([
    torchvision.transforms.Resize(576),
    torchvision.transforms.CenterCrop(576),
    torchvision.transforms.ToTensor()
])

trans_mask = torchvision.transforms.Compose([
    torchvision.transforms.Resize(576, interpolation=Image.NEAREST),
    torchvision.transforms.CenterCrop(576),
    torchvision.transforms.PILToTensor(),
    lambda x: x.squeeze().long(),
    lambda x: torch.where(x == 3, torch.tensor(1).long(), x)
])

# ------------------------- 数据集加载 -------------------------
train_dataset = torchvision.datasets.OxfordIIITPet(
    root='/home/RAID0/wtx/project/U-Net/dataset',
    split='trainval',
    target_types='segmentation',
    transform=trans_totensor,
    target_transform=trans_mask,
    download=True
)

train_sampler = DistributedSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

# ------------------------- 模型初始化 -------------------------
unet = Unet(num_class=3).to(device)
unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank])

# ------------------------- 训练循环 -------------------------
optim = Adam(unet.parameters())
loss = nn.CrossEntropyLoss().to(device)

# 只在主进程创建SummaryWriter
if dist.get_rank() == 0:
    writer = SummaryWriter('/home/RAID0/wtx/project/U-Net/train_model')
else:
    writer = None

total_train_step = 0
for epoch in range(200):
    train_sampler.set_epoch(epoch)  # 设置epoch保证shuffle正确
    unet.train()

    epoch_loss = 0
    for data in train_dataloader:
        optim.zero_grad()
        image, target = data
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = unet(image)

        # 验证标签范围
        assert (target >= 0).all(), "标签包含负数!"
        assert (target < output.shape[1]).all(), "标签越界!"

        result_loss = loss(output, target)
        result_loss.backward()
        optim.step()

        epoch_loss += result_loss.item()
        total_train_step += 1

        if total_train_step % 100 == 0 and dist.get_rank() == 0:
            print(f"训练次数：{total_train_step}, Loss: {result_loss.item():.4f}")
            writer.add_scalar('训练损失', result_loss.item(), total_train_step)

    # 只在主进程保存模型和输出日志
    if dist.get_rank() == 0:
        writer.add_scalar('每个epoch损失总和', epoch_loss, epoch)
        print(f'--------------------------第{epoch + 1}轮训练已完成！--------------------------')
        print(f'epoch：{epoch + 1}, epoch_loss：{epoch_loss:.4f}')

        torch.save(unet.module.state_dict(),  # 注意使用.module获取原始模型
                   f'/home/RAID0/wtx/project/U-Net/model/model_{epoch + 1}.pth')
        print(f'第{epoch + 1}轮训练模型已保存')

if dist.get_rank() == 0:
    writer.close()
dist.destroy_process_group()