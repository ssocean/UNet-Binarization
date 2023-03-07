import cv2
import numpy as np

import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils
from workshop.GeneralTools import *


def otsu_bin(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res


def overlapping_seg(img_path: str, rstdir_pth: str, patch_h: int = 256, patch_w: int = 256, stride_h: int = 128,
                    stride_w: int = 128, suffix: str = None):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param rstdir_pth: 切片存放路径
    :param stride_w: 横方向步长，默认128
    :param stride_h: 纵方向步长，默认128
    :param patch_w: 切片宽，默认256
    :param patch_h: 切片高，默认256
    :param suffix: 切片后缀，默认于原图相等
    :return:切片总数
    '''
    img_info = os.path.split(img_path)[1].split('.')
    img_name = img_info[0]
    img_suffix = img_info[1]
    if suffix:  # 如果给了后缀
        img_suffix = suffix

    img = cv2.imread(img_path)
    h, w, c = img.shape

    n_w = int((w - patch_w) / stride_w) * stride_w + patch_w
    n_h = int((h - patch_h) / stride_h) * stride_h + patch_h

    img = cv2.resize(img, (n_w, n_h))
    n_patch_h = (h - patch_h) // stride_h + 1
    n_patch_w = (w - patch_w) // stride_w + 1
    n_patches = n_patch_h * n_patch_w
    auto_make_directory(rstdir_pth)
    for i in range(n_patch_w):
        for j in range(n_patch_h):
            y1 = j * stride_h
            y2 = y1 + patch_h
            x1 = i * stride_w
            x2 = x1 + patch_w
            roi = img[y1:y2, x1:x2]
            retval = cv2.imwrite(fr"{rstdir_pth}/{img_name}_{str(i)}_{str(j)}.{img_suffix}", roi)
            assert retval, r"image saved failure"
    return n_patches


def showim(img:np.ndarray):
    '''
    展示图片
    :param img: ndarray格式的图片
    '''
    cv2.namedWindow(f'image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f'image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def labelme_to_dataset(dir,output_dir):
    '''
    预先准备
    1. 将json文件全部考入一个文件夹内
    2. 装有labelme环境的prompt cd指令进入该文件夹 输入
       for /r %i in (*) do labelme_json_to_dataset %i
       注意观察文件数量，该命令不会自行终止

    '''
    auto_make_directory(output_dir)
    dir_list = get_dirs_pth(dir)
    dir_name_list = get_dirs_name(dir)
    assert len(dir_list)==len(dir_name_list), "长度不一致"
    for i in range(len(dir_list)):
        img = cv2.imread(os.path.join(dir_list[i],'label.png'),1)#彩色读图片

        img = otsu_bin(img)
        img_name = dir_name_list[i].replace('_json','')+'.png'
        cv2.imwrite(os.path.join(output_dir,img_name),img)
        # showim(img)
    pass

labelme_to_dataset(r'D:\hongpu\json',r'D:\hongpu\mask')
