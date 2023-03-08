import os
import random
from os import listdir
from os.path import splitext
from tqdm import tqdm
import numpy as np
import cv2
from utils.FileOperator import *



# img_tf_flip(r'E:\Project\Unet-vanilla\data\img_backup',r'E:\Project\Unet-vanilla\data\mask_backup')

# img_tf_flip(r'../data/backup/img', r'../data/backup/mask')
def dir_bit_or(img_dir, mask_dir,dst_dir):
    '''
    将CPU生成图所在文件夹与GPU生成图所在文件夹中所有的图对应地进行按位或操作
    :param img_dir: CPU生成图路径
    :param mask_dir: GPU生成图路径
    :return:
    '''
    img_ids = get_files_name(img_dir)
    mask_ids = get_files_name(mask_dir)
    length = len(img_ids)
    for i in tqdm(range(0, length)):
        img = cv2.imread(fr'{img_dir}/{img_ids[i]}.png')
        t,img = cv2.threshold(img,210,255,cv2.THRESH_BINARY)
        mask = cv2.imread(fr'{mask_dir}/{mask_ids[i]}.png')
        dst = cv2.bitwise_or(img, mask)
        pth = os.path.join(dst_dir,f'{img_ids[i]}_bitor.png')
        ret = cv2.imwrite(pth, dst)
        assert ret, 'save failed'


def copy_mask():
    '''
    复制mask
    :return:
    '''
    img_ids = get_files_name(r'..\data\masks-backup')
    length = len(img_ids)
    for i in tqdm(range(0, length)):
        # print(img_ids[i])
        img = cv2.imread(fr'../data/masks-backup/{img_ids[i]}.png')
        ret = cv2.imwrite(fr'../data/masks/{img_ids[i]}_scratch.png', img)
        assert ret, 'save failed'
        ret = cv2.imwrite(fr'../data/masks/{img_ids[i]}_stain.png', img)
        assert ret, 'save failed'
        ret = cv2.imwrite(fr'../data/masks/{img_ids[i]}_dot.png', img)
        # img = cv2.imread(fr'../data/masks-backup/background.png')
        # ret = cv2.imwrite(fr'../data/temp/{i}_background.png', img)
        assert ret, 'save failed'
        # print(i)
    print('done')


import PIL #'6.2.1'
import cv2 #'4.1.1'

def patchit(root_dir,dst_dir):
    auto_make_directory(dst_dir)
    file_paths = get_files_pth(root_dir)
    for pth in file_paths:
        img = cv2.imread(pth,0)

        auto_make_directory(dst_dir)
        _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(dst_dir,os.path.basename(pth)),img, [cv2.IMWRITE_PNG_BILEVEL, 1])


