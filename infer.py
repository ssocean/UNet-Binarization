import argparse
from os.path import splitext
from datasets.BinarizationDataset import Single_Img_Infer_Dataset

import cv2
import torch
import numpy as np

from tqdm import tqdm

import models.Models as Models
from models.Models import UNet, AttU_Net
import os
import glob
from torch.utils.data import DataLoader
from utils.FileOperator import *
import torchvision.transforms as transforms
parser = argparse.ArgumentParser()
parser.add_argument("--imgs_dir",default=r'D:\Data\DAR\image')
parser.add_argument("--out_dir",default=r'D:\Data\DIBCO-mini\out')
parser.add_argument("--model_pth",default=r'C:\Users\Ocean\Documents\GitHub\outputs\2023_Mar_08_13_UNet_BestResult.pth')
parser.add_argument("--batch_size",default=4)
parser.add_argument("--patch_size",default=256)
parser.add_argument("--bitwise_img_size",default=1024,help='img size for bitwise ot operations. We recommand you to set this args as large as possible.')

args = parser.parse_args()
def fold32(tup):
    return (int(int(tup[0] / 32) * 32), int(int(tup[1] / 32) * 32))


def _single_pred(img_np, net, device, is_vis=True, is_normalize=True, is_binarize=False):
    """
    单个图片的预测
    :param img_np: 图片地址，包含后缀
    :param is_vis: 布尔值默认真，是否查看结果
    :param is_normalize: 布尔值默认真，是否归一化
    :param is_binarize: 布尔值默认假，是否二值化
    :return:
    """
    # img = cv2.imread(img_np)
    img_tensor = torch.from_numpy(img_np)  # 转tensor
    img_tensor = img_tensor.unsqueeze(0).transpose(3, 1).transpose(2, 3)  # 转维度

    # print(img_tensor.size())

    img_tensor = img_tensor.to(device=device, dtype=torch.float32)  # 转设备、类型
    # print(img_tensor.shape)
    mask_pred = net(img_tensor)
    cpu_tensor = mask_pred.cpu()
    pred_np = cpu_tensor.detach().numpy()  # 转回numpy
    pred_np = np.squeeze(pred_np, 0)
    pred_np = np.swapaxes(pred_np, 0, 2)
    pred_np = np.swapaxes(pred_np, 0, 1)
    # print(pred_np)
    # 归一化
    if is_normalize:
        _dst = np.zeros((1, 256, 256))
        pred_np = cv2.normalize(pred_np, _dst, 0, 255, cv2.NORM_MINMAX)
        pred_np = pred_np.astype(np.uint8)

    # 二值化
    if is_binarize:
        pred_np = pred_np.astype(np.uint8)
        _, pred_np = cv2.threshold(pred_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # print(pred_np)
    if is_vis:
        cv2.namedWindow(f'img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f'img', pred_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return pred_np


imgs_dir = 'data/imgs/'


def _generate_rst_img(img_path, rst_dir, net, device, patch_size=256, mag_scale=1.0, is_bin=False):
    '''
        生成二值图
        :param img_path: 图片路径
        :param rst_dir: 结果图存放路径
        :param net: 网络模型实例
        :param patch_size: 切片大小，默认256，如果为0时不进行切分
        :param mag_scale: 单切片放缩比例，默认1.0
        :param device: 使用是被，默认为cpu
        :param is_bin: 是否输出二值图，默认否
        :return:
        '''

    fname = os.path.basename(img_path)
    fname = '.'.join(fname.split('.')[:-1])

    net = net
    net.to(device)

    img = cv2.imread(img_path)
    (h, w) = img.shape[:2]  # 保存图像的形状(h,w,c)

    # 预先定义存放预测结果的numpy数组
    if patch_size == 0:  # 如果整张不切分
        roi = img
        roi = cv2.resize(roi, (int(int(mag_scale * w) / 32) * 32, int(int(mag_scale * h) / 32) * 32))
        rst = _single_pred(roi, net, device, is_vis=False, is_binarize=False)
        pred_np = cv2.resize(rst, (w, h))
    else:  # 否则切分
        n_w = (int(w / patch_size) + 1) * patch_size
        n_h = (int(h / patch_size) + 1) * patch_size
        count_w = int(w / patch_size) + 1
        count_h = int(h / patch_size) + 1
        img = cv2.resize(img, (n_w, n_h))
        pred_np = np.zeros((n_h, n_w), dtype=np.uint8)
        for i in range(0, count_w):
            for j in range(0, count_h):
                roi = img[j * patch_size:(j + 1) * patch_size,
                      i * patch_size:(i + 1) * patch_size]  # roi为每一个小切片，region of interest
                # roi = cv2.resize(roi, (int(mag_scale * patch_size), int(mag_scale * patch_size)))  # 对roi进行放缩
                roi = cv2.resize(roi, fold32((int(mag_scale * patch_size), int(mag_scale * patch_size))))
                rst = _single_pred(roi, net, device, is_vis=False, is_binarize=False)  # 获得单roi的预测结果
                # print(rst)
                rst = cv2.resize(rst, (patch_size, patch_size))  # 将roi缩放回原尺寸

                pred_np[j * patch_size:(j + 1) * patch_size,
                i * patch_size:(i + 1) * patch_size] = rst  # roi加入预先定义存放预测结果的numpy数组

        pred_np = cv2.resize(pred_np, (w, h))  # 预先定义存放预测结果的numpy数组放缩至图像原尺寸
    if is_bin:
        # pred_np = cv2.cvtColor(pred_np, cv2.COLOR_BGR2GRAY)
        _, pred_np = cv2.threshold(pred_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not os.path.exists(rst_dir):  # 如果目标路径不存在，创建该路径
        os.makedirs(rst_dir)
    file_name = f'{fname}_{patch_size}_{mag_scale}_{net.name}.png'
    retval_pth = os.path.join(rst_dir, file_name)

    retval = cv2.imwrite(retval_pth, pred_np)
    assert retval, r"预测结果保存失败"


def _gray_to_bin(img_path, rst_dir, patch_size=256, mag_scale=1.0):
    '''
    灰度转二值
    :param img_path: 图片路径
    :param rst_dir: 结果图存放路径
    :param net: 网络模型实例
    :param patch_size: 切片大小，默认256，如果为0时不进行切分
    :param mag_scale: 单切片放缩比例，默认1.0
    :param device: 使用是被，默认为cpu
    :param is_bin: 是否输出二值图，默认否
    :return:
    '''

    fname = os.path.basename(img_path)
    fname = '.'.join(fname.split('.')[:-1])

    img = cv2.imread(img_path, 0)
    (h, w) = img.shape[:2]  # 保存图像的形状(h,w,c)

    n_w = (int(w / patch_size) + 1) * patch_size
    n_h = (int(h / patch_size) + 1) * patch_size
    count_w = int(w / patch_size) + 1
    count_h = int(h / patch_size) + 1
    img = cv2.resize(img, (n_w, n_h))
    pred_np = np.zeros((n_h, n_w), dtype=np.uint8)

    for i in range(0, count_w):
        for j in range(0, count_h):
            roi = img[j * patch_size:(j + 1) * patch_size,
                  i * patch_size:(i + 1) * patch_size]  # roi为每一个小切片，region of interest
            roi = cv2.resize(roi, (int(mag_scale * patch_size), int(mag_scale * patch_size)))  # 对roi进行放缩
            roi = roi.astype(np.uint8)
            _, rst = cv2.threshold(roi, 205, 255, cv2.THRESH_BINARY)
            rst = cv2.resize(rst, (patch_size, patch_size))  # 将roi缩放回原尺寸

            pred_np[j * patch_size:(j + 1) * patch_size,
            i * patch_size:(i + 1) * patch_size] = rst  # roi加入预先定义存放预测结果的numpy数组

    pred_np = cv2.resize(pred_np, (w, h))  # 预先定义存放预测结果的numpy数组放缩至图像原尺寸

    if not os.path.exists(rst_dir):  # 如果目标路径不存在，创建该路径
        os.makedirs(rst_dir)
    file_name = f'{fname}_{patch_size}_{mag_scale}.png'
    retval_pth = os.path.join(rst_dir, file_name)

    retval = cv2.imwrite(retval_pth, pred_np)
    assert retval, r"预测结果保存失败"


def getFileList(pth):
    rst = []
    glob_pth = os.path.join(pth, '*.jpg')
    for filename in glob.glob(glob_pth):
        rst.append(filename)
    return rst

import torchvision.utils
def dir_c2b(img_dir, rst_dir, need_bitwise=True):
    '''
    将一个文件夹内所有的彩色图片进行二值化
    :param img_dir: 待二值化图片所在路径
    :param rst_dir: 输出路径
    :param is_bin: 是否二值化，默认否、输出灰度图
    :return:
    '''
    auto_make_directory(rst_dir)
    # 保存的模型文件路径
    net = Models.UNet()
    assert args.bitwise_img_size%32==0,'bitwise_img_size should only be a multiple of 32' 
    weights_path = args.model_pth # 保存的模型文件路径

    net.load_state_dict(torch.load(weights_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net.to(device=device)
    net.eval()
    
    print(img_dir)
    imgs_list = get_files_pth(img_dir)
    print(imgs_list)
    import math
    from PIL import Image
    for img_pth in tqdm(imgs_list):
        img = Image.open(img_pth)
        ori_w,ori_h = img.size
        target_size = math.floor(img.size[0] / args.patch_size) * args.patch_size, math.floor(img.size[1] / args.patch_size) * args.patch_size
        nrow = int(target_size[0] / args.patch_size)

        big_pic = img
        pred_np = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        
        transform_infer = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        ])
        
        img = transform_infer(img)
        
        single_dataset = Single_Img_Infer_Dataset(img,patch_size=args.patch_size)
        
        data_loader = DataLoader(single_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        pred_lst = []
        
        for batch in data_loader:
            batch = batch.to(device=device)
            with torch.no_grad():
                out = net(batch)
            pred_lst.append(out)
        rst_batch = torch.cat(pred_lst, dim=0)
        grid = torchvision.utils.make_grid(rst_batch, nrow=nrow, padding=0)
        
        img_name = get_filename_from_pth(img_pth)
        
        pred_np = torch.einsum("chw->hwc", grid).cpu().detach().numpy()
        pred_np = cv2.resize(pred_np,(ori_w,ori_h))
        pred_np *= 255
        pred_np = pred_np.astype(np.uint8)
       
        
        pred_np = cv2.cvtColor(pred_np, cv2.COLOR_BGR2GRAY)
        _, pred_np = cv2.threshold(pred_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        transform = transforms.Compose([
            transforms.Resize((args.bitwise_img_size,args.bitwise_img_size)), #长边缩放至1024
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])    
        
        
        if need_bitwise:
            big_pic =transform(big_pic).unsqueeze(0).to(device=device)
            with torch.no_grad():
                out = net(big_pic)
            out = torch.einsum('bchw->bhwc', out).cpu().detach().numpy()
            big_pic_np = np.uint8(out[0])
            big_pic_np = cv2.resize(big_pic_np,(ori_w,ori_h))
            _, big_pic_np = cv2.threshold(big_pic_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            pred_np = cv2.bitwise_or(big_pic_np,pred_np)
            
        
        ret = cv2.imwrite(os.path.join(out_dir,f"{img_name}.png"), pred_np)
        assert ret, 'Save Failed'
        


if __name__ == '__main__':
    imgs_dir = args.imgs_dir
    out_dir = args.out_dir
    dir_c2b(imgs_dir, out_dir, True)