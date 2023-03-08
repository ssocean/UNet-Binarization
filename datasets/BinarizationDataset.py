from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
from utils.FileOperator import *
import torch
import torch.nn.functional as F
from PIL import Image

class BinDataset(Dataset):
    '''
    数据集加载类
    '''

    def __init__(self, imgs_dir, masks_dir,transform=None):
        """
        在此完成数据集的读取
        :param imgs_dir: 图片路径,末尾需要带斜杠
        :param masks_dir: mask路径，末尾需要带斜杠
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.imgs = get_files_pth(imgs_dir)
        self.masks = get_files_pth(masks_dir)
        assert len(self.imgs) == len(self.masks) and len(self.imgs)>=1, 'Number of input image is expected to be the same as the number of mask'
        self.transform = transform



    def __len__(self):
        '''
        返回数据集中包含的样本个数
        :return: 数据集中包含的样本个数
        '''
        return len(self.imgs)

    def __getitem__(self, item):
        '''
        根据item，返回图片和它对应的标注图片
        :param item: 框架指定，请勿修改
        :return: 字典{'img':FloatTensor类型,'mask'：FloatTensor类型}
        '''

            
        img = cv2.imread(self.imgs[item], 1)
        mask = cv2.imread(self.masks[item], 1)

        assert img.size == mask.size, \
            f'图片与掩膜 {self.imgs[item]} 大小不一致,图片： {img.size} 掩膜： {mask.size}'

        _,mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # 数据问题 需要先做一次二值化
        # mask = mask/255
        
        mask = Image.fromarray(mask)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
            
        return {
            'image': img,
            'mask': mask
        }
        pass


class Single_Img_Infer_Dataset(Dataset):
    '''
    数据集加载类
    '''

    def __init__(self, img, patch_size):
        """
        在此完成数据集的读取
        :param imgs_dir: 图片路径,末尾需要带斜杠
        :param masks_dir: mask路径，末尾需要带斜杠
        """
        self.img = img
        self.patch_size = patch_size
        self.blocks = self.patchify()
    def patchify(self):
        
        
        # 将图像展开为一个二维矩阵
        unfold = F.unfold(self.img.unsqueeze(0), kernel_size=self.patch_size, stride=self.patch_size)
        
        # 将展开后的矩阵转置为(batch_size, num_blocks, block_size^2)
        unfold = unfold.transpose(1, 2)
        
        # 将展开后的矩阵转换为图像
        blocks = unfold.view(-1, self.img.shape[0], self.patch_size, self.patch_size) 
        return blocks


    def __len__(self):
        '''
        返回数据集中包含的样本个数
        :return: 数据集中包含的样本个数
        '''
        return self.blocks.shape[0]

    def __getitem__(self, item):
        '''
        根据item，返回图片和它对应的标注图片
        :param item: 框架指定，请勿修改
        :return: 字典{'img':FloatTensor类型,'mask'：FloatTensor类型}
        '''
        return self.blocks[item,:,:,:]
