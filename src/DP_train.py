import torch.nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model import *
import time

start_time = time.time()

writer = SummaryWriter('/home/RAID0/wtx/project/U-Net/DP_train_model')
device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

# ------------------------- 数据预处理修正 -------------------------
trans_totensor = torchvision.transforms.Compose([
    torchvision.transforms.Resize(576),  # 572 -> 576
    torchvision.transforms.CenterCrop(576),  # 原数据集中有非正方形，强制转换为正方形
    torchvision.transforms.ToTensor()
])

trans_mask = torchvision.transforms.Compose([
    torchvision.transforms.Resize(576, interpolation=Image.NEAREST),
    torchvision.transforms.CenterCrop(576),
    torchvision.transforms.PILToTensor(),
    lambda x: x.squeeze().long(),
    # 关键修正：将标签中的3映射为1（根据OxfordIIITPet数据集特性）
    lambda x: torch.where(x == 3, torch.tensor(1).long(), x)
])

# ------------------------- 数据集加载 -------------------------
train_dataset = torchvision.datasets.OxfordIIITPet(root='/home/RAID0/wtx/project/U-Net/dataset', split='trainval',
                                                   target_types='segmentation', transform=trans_totensor,
                                                   target_transform=trans_mask, download=True)
# test_dataset = torchvision.datasets.OxfordIIITPet(root= 'C:/Users/45730/Desktop/U-Net/dataset',split='test',target_types ='segmentation',transform= trans_totensor,target_transform=ttrans_mask,download=True)

train_dataloader = DataLoader(train_dataset, 8, True)
# test_dataloader = DataLoader(test_dataset,1,True)

# ------------------------- 模型初始化、数据并行 -------------------------
# 关键点：输出通道数设为3（对应修正后的标签类别0、1、2）
unet = Unet(num_class=3)  # 修改此处为3类
unet = torch.nn.DataParallel(module=unet, device_ids=[1, 0, 2, 3])
unet.to(device)
# ------------------------- 训练循环 -------------------------
optim = Adam(unet.parameters())
loss = nn.CrossEntropyLoss().to(device)

# 记录最优模型和对应损失
best_epoch = 0
best_epoch_loss = 1000

total_train_step = 0
for epoch in range(200):
    epoch_loss = 0
    for data in train_dataloader:
        optim.zero_grad()
        image, target = data
        image = image.to(device)
        target = target.to(device)
        output = unet(image)

        # 验证标签范围（调试用）
        # print("模型输出的类别数:", output.shape[1])  # 应为3
        # print("标签最大值:", target.max().item())  # 应<=2
        # 断言检查（可选）
        assert (target >= 0).all(), "标签包含负数!"
        assert (target < output.shape[1]).all(), "标签越界!"

        # 计算损失
        result_loss = loss(output, target)
        result_loss.backward()
        optim.step()

        epoch_loss += result_loss.item()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {result_loss.item():.4f}")
            writer.add_scalar('训练损失', result_loss.item(), total_train_step)

    writer.add_scalar('每个epoc损失总和', epoch_loss, epoch)
    print(f'--------------------------第{epoch + 1}轮训练已完成！--------------------------')
    print(f'epoch：{epoch + 1},epoch_loss：{epoch_loss:.4f}')

    torch.save(unet.state_dict(), f'/home/RAID0/wtx/project/U-Net/DP_model/model_{epoch + 1}.pth')
    print(f'第{epoch + 1}轮训练模型已保存')

    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        best_epoch = epoch + 1

writer.close()

end_time = time.time()

print(f'总运行时间：{end_time - start_time}')
print(f'最好模型为第{best_epoch}次训练模型，对应损失为{best_epoch_loss}')
