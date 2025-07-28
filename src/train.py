from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
from datasets.dataset import NPY_datasets
from configs.config_setting import setting_config
from utils import *
import time

config = setting_config()

writer = SummaryWriter('/home/wtx/project/U-Net-skin/train_model')

print('#----------GPU init----------#')
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print('#----------Preparing dataset----------#')
train_dataset = NPY_datasets(config.data_path, config, train=True)
train_dataloader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=config.num_workers)
val_dataset = NPY_datasets(config.data_path, config, train=False)
val_dataloader = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=config.num_workers,
                        drop_last=True)

print('#----------Prepareing Model----------#')
unet = Unet(1)
unet.to(device)
unet = torch.nn.DataParallel(module=unet, device_ids=[0, 1, 2, 3])

print('#----------Prepareing loss, opt, sch and amp----------#')
loss = config.criterion
optim = get_optimizer(config,unet)
sch = get_scheduler(config,optim)

print('#----------Set other params----------#')
total_train_step = 0
# 记录最优模型和对应损失
best_epoch = 0
best_epoch_loss = 1000
total_train_time = 0

print('#----------Training----------#')
for epoch in range(300):
    print(f'--------------------------第{epoch + 1}轮训练开始：--------------------------')
    epoch_start_time = time.time()
    epoch_loss = 0
    for data in train_dataloader:
        optim.zero_grad()
        image, target = data
        image = image.to(device)
        target = target.to(device)
        image = image.float()
        output = unet(image)
        result_loss = loss(output, target)
        result_loss.backward()
        optim.step()
        epoch_loss += result_loss.item()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {result_loss.item():.4f}")
            writer.add_scalar('训练损失', result_loss.item(), total_train_step)

    writer.add_scalar('每个epoc损失总和', epoch_loss, epoch)
    print(f'--------------------------第{epoch + 1}轮训练已完成！--------------------------')
    print(f'epoch：{epoch + 1},epoch_loss：{epoch_loss:.4f}')

    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        best_epoch = epoch + 1

    epoch_end_time = time.time()
    total_train_time = epoch_end_time - epoch_start_time

    torch.save(unet.state_dict(), f'/home/wtx/project/U-Net-skin/model_save/model_{epoch + 1}.pth')
    print(f'第{epoch + 1}轮训练模型已保存，本轮训练用时为{(epoch_end_time-epoch_start_time):.4f}')

writer.close()

print(f'总运行时间：{total_train_time}')
print(f'最好模型为第{best_epoch}次训练模型，对应损失为{best_epoch_loss}')
