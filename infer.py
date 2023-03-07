import argparse
from os.path import splitext

import cv2
import torch
import numpy as np

from tqdm import tqdm

import models.Models as Models
from models.Models import UNet, AttU_Net
import os
import glob

from workshop.GeneralTools import auto_make_directory

parser = argparse.ArgumentParser()
parser.add_argument("--imgs_dir")
parser.add_argument("--out_dir")
parser.add_argument("--model_pth")
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


def dir_c2b(img_dir, rst_dir, is_bin=False):
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

    weights_path = args.model_pth # 保存的模型文件路径

    net.load_state_dict(torch.load(weights_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(img_dir)
    imgs_list = getFileList(img_dir)
    print(imgs_list)
    for img_name in tqdm(imgs_list):
        pth = os.path.join(img_dir, img_name)
        _generate_rst_img(pth, rst_dir, net, device, patch_size=0, mag_scale=1.0, is_bin=is_bin)


# /mnt/penghai/data/img/ /mnt/penghai/hp-unet/unet-rst/img


if __name__ == '__main__':
    imgs_dir = args.imgs_dir
    out_dir = args.out_dir
    dir_c2b(imgs_dir, out_dir, True)