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


def init_logger():
    '''
    初始化日志类
    :return: 日志类实例对象
    '''
    # 日志模块
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    os.makedirs('logs/',exist_ok=True)
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
          scheduler,
          writter,
          epochs=1):


    # print('数据集加载完毕')
    global_step = 0
    
    logger.info('开始训练，即将读取epoch')
    # p = trained_epoch + 1
    min_loss = 1.0
    start_time = time.localtime()
    
    for epoch in tqdm(range(epochs)):
        logger.info(f'--------------------------第{epoch}轮训练开始--------------------------')

        net.train()
        epoch_loss = 0

        e_times = 1
        epoch_fm = 0.0
        for batch in tqdm(train_loader):

            e_times = e_times + 1
            
            imgs = batch['image'].to(device=device)
            true_masks = batch['mask'].to(device=device)

            masks_pred = net(imgs)

            loss = criterion(masks_pred, true_masks)
            writter.add_images('IMG', imgs, global_step=global_step,  dataformats='NCHW')
            writter.add_images('MASK', true_masks, global_step=global_step,  dataformats='NCHW')
            writter.add_images('PRED', masks_pred, global_step=global_step,  dataformats='NCHW')
            writter.add_scalar('loss', loss, global_step=global_step)
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
                logger.info(f'{epoch}轮第{e_times}个loss:' + str(loss_float))
            # epoch_fm += train_fm

        val_score, fmeasure = eval_fm(net, val_loader, device)
        writter.add_scalar('eval_fm', fmeasure, global_step=global_step)
        scheduler.step(val_score)
        logger.info(f'{epoch}轮训练集FM:{str(round(epoch_fm / len(train_loader), 2))},验证集FM:' + str(fmeasure))
        if epoch % args.save_ckpt_every_epoch == 0:  
            
            os.makedirs(args.dir_checkpoint,exist_ok=True)
            torch.save(net.state_dict(),
                        args.dir_checkpoint + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_{net.name}_AUTO{epoch}.pth')
            logger.info(
                f'{device}下的{net.name}网络第{epoch}轮结果被保存为：' + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_AUTO{epoch}.pth')
        # 保存最优
        if loss_float < min_loss:
            os.makedirs(args.dir_checkpoint,exist_ok=True)
            torch.save(net.state_dict(),
                        args.dir_checkpoint + f'{time.strftime("%Y_%b_%d_%H", start_time)}_{net.name}_BestResult.pth')
            logger.info(
                f'{device}下的{net.name}网络最优结果被保存为：' + f'{time.strftime("%Y_%b_%d_%H", start_time)}_BestResult.pth')


def init_tensorboard(out_dir: str = 'tb-logs/'):
    os.makedirs(out_dir,exist_ok=True)

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
        imgs_dir = r'D:\Data\DIBCO-mini\img/'
        masks_dir = r'D:\Data\DIBCO-mini\gt/'

    # 网络模型保存路径
    else:
        imgs_dir = args.imgs_dir
        masks_dir = args.masks_dir

    writter = init_tensorboard()

    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络
    net = Models.UNet()

    net.to(device=device)
    
    # 当输入尺寸较为固定，使用benchmark加速网络训练
    cudnn.benchmark = True
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),])
        # transforms.Normalize(mean=[0.5], std=[0.229, 0.224, 0.225])
    dataset = BinarizationDataset.BinDataset(imgs_dir, masks_dir,transform=transform_train)
    
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    criterion = BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    try:
        train(net=net,
              optimizer=optimizer,
              train_loader=train_loader,
              val_loader=val_loader,
              criterion=criterion,
              epochs=args.epoch,
              device=device,
              scheduler=scheduler,
              writter = writter,
              )
        
    except KeyboardInterrupt:
        writter.close()
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    writter.close()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgs_dir",default=None,help='If you dont want to use the command line, leave all parameters blank & jsut python train.py.')
    parser.add_argument("--masks_dir",default=None)
    parser.add_argument("--dir_checkpoint",default=r'outputs/')


    parser.add_argument("--input_size",default=256)
    parser.add_argument("--epoch",default=200)
    parser.add_argument("--batch_size",default=4)
    
    parser.add_argument("--val_percent",default=0.1)
    
    parser.add_argument("--lr",default=0.001)

    parser.add_argument("--weight_decay",default=1e-8)
    parser.add_argument("--momentum",default=0.9)

    
    parser.add_argument("--save_ckpt_every_epoch",default=20)
    args = parser.parse_args()

    logger = init_logger()

    main(args)




