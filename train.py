import argparse
import time

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import models.Models as Models
import datasets.BinarizationDataset as BinarizationDataset
import os
import sys
import torch.nn as nn
from torch import optim
import logging
from tqdm import tqdm
from utils.eval import *
import torch
from torch.nn import BCEWithLogitsLoss
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def tensor_to_ndarray(t):
    cpu_tensor = t.cpu()
    res = cpu_tensor.detach().numpy()  # 转回numpy
    # print(res.shape)
    res = np.squeeze(res, 1)
    # res = np.swapaxes(res, 0, 2)
    # res = np.swapaxes(res, 0, 1)
    return res


def init_logger():
    '''
    初始化日志类
    :return: 日志类实例对象
    '''
    # 日志模块
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(fr'logs/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 输出到日志
    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger.info("Start print log") #一般信息
    logger.debug("Do something") #调试显示
    logger.warning("Something maybe fail.")#警告
    logger.info("Finish")
    '''
    return logger


def train(net,
          device,
          train_loader,
          val_loader,
          optimizer,
          criterion,
          epochs=1,
          lr=0.001,
          trained_epoch=0):


    # print('数据集加载完毕')
    global_step = 0
    

    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    
    # pixelwise_loss = nn.SmoothL1Loss(reduction='mean')
    # criterion = nn.BCELoss()
    logger.info('开始训练，即将读取epoch')
    p = trained_epoch + 1
    min_loss = 1.0
    start_time = time.localtime()
    try:
        for epoch in tqdm(range(epochs)):
            # logger.info('开始训练，即将读取epoch')
            # print('/r')
            logger.info(f'--------------------------第{p}轮训练开始--------------------------')

            net.train()
            epoch_loss = 0
            # print(f"第{epochs}轮开始训练")
            e_times = 1
            epoch_fm = 0.0
            for batch in tqdm(train_loader):
                # logger.info(f'第{p}轮开始训练第{e_times}个batch')
                e_times = e_times + 1
                
                imgs = batch['image'].to(device=device)
                true_masks = batch['mask'].to(device=device)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
    
                optimizer.zero_grad()
                loss.backward()
                
                 # 梯度裁剪 防止梯度爆炸
                nn.utils.clip_grad_value_(net.parameters(), 0.1) 
                
                optimizer.step()
                
                global_step += 1
                loss_float = float(loss)
                # train_fm = fm(masks_pred, true_masks)
                if e_times%5 == 0:
                    logger.info(f'{p}轮第{e_times}个loss:' + str(loss_float))
                # epoch_fm += train_fm

            val_score, fmeasure = eval_fm(net, val_loader, device)
            # scheduler.step(val_score)
            # logger.info(f'{p}轮训练集FM:{str(round(epoch_fm / len(train_loader), 2))},验证集FM:' + str(fmeasure))
            if p % args.save_ckpt_every_epoch == 0:  
                
                os.mkdir(args.dir_checkpoint,exist_ok=True)
                torch.save(net.state_dict(),
                           args.dir_checkpoint + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_{net.name}_AUTO{p}.pth')
                logger.info(
                    f'{device}下的{net.name}网络第{p}轮结果被保存为：' + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_AUTO{p}.pth')
            # 保存最优
            if loss_float < min_loss:
                os.mkdir(args.dir_checkpoint,exist_ok=True)
                torch.save(net.state_dict(),
                           args.dir_checkpoint + f'{time.strftime("%Y_%b_%d_%H", start_time)}_{net.name}_BestResult.pth')
                logger.info(
                    f'{device}下的{net.name}网络最优结果被保存为：' + f'{time.strftime("%Y_%b_%d_%H", start_time)}_BestResult.pth')
            p = p + 1
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def init_tensorboard(out_dir: str = 'logs'):
    if not os.path.exists(out_dir):  ##目录存在，返回为真
        os.makedirs(out_dir)

    writer = SummaryWriter(log_dir=out_dir)
    '''
    https://pytorch.org/docs/stable/tensorboard.html
    writer.
    add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
    add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
    add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
    '''

    #  writer.close()  需在最后关闭
    return writer

def main(args):
        # # 标注图像存放路径
    if args.imgs_dir is None:
    # masks_dir = args.masks_dir
        imgs_dir = r'/opt/data/private/data/gray-image/'
        masks_dir = r'/opt/data/private/data/mask/'
        dir_checkpoint = r'weights/'

    # 网络模型保存路径
    else:
        imgs_dir = args.imgs_dir
        masks_dir = args.masks_dir
        dir_checkpoint = args.dir_checkpoint



    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络
    net = Models.UNet()

    net.to(device=device)
    
    # 当输入尺寸较为固定，使用benchmark加速网络训练
    cudnn.benchmark = True
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = BinarizationDataset.BinDataset(imgs_dir, masks_dir,transform=transform_train)
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    criterion = BCEWithLogitsLoss()
    try:
        train(net=net,
              optimizer=optimizer,
              train_loader=train_loader,
              criterion=criterion,
              epochs=args.epoch,
              batch_size=args.batch_size,
              lr=args.lr,
              device=device,
              val_percent=args.val_percent,
              trained_epoch=0
              )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgs_dir",default=None,help='If you dont want to use the command line, leave all parameters blank.')
    parser.add_argument("--masks_dir",default=None)
    parser.add_argument("--dir_checkpoint",default=r'outputs/')


    parser.add_argument("--input_size",default=256)
    parser.add_argument("--epoch",default=200)
    parser.add_argument("--batch_size",default=32)
    
    parser.add_argument("--val_percent",default=0.1)
    
    parser.add_argument("--lr",default=0.001)

    parser.add_argument("--weight_decay",default=1e-8)
    parser.add_argument("--val_percent",default=0.9)

    
    parser.add_argument("--save_ckpt_every_epoch",default=20)
    args = parser.parse_args()

    logger = init_logger()

    main(args)




