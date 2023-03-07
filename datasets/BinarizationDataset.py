from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import cv2


class BinDataset(Dataset):
    '''
    数据集加载类
    '''

    def __init__(self, imgs_dir, masks_dir):
        """
        在此完成数据集的读取
        :param imgs_dir: 图片路径,末尾需要带斜杠
        :param masks_dir: mask路径，末尾需要带斜杠
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)  # 获取图片名称，ids是一个列表
                    if not file.startswith('.')]
        pass



    def __len__(self):
        '''
        返回数据集中包含的样本个数
        :return: 数据集中包含的样本个数
        '''
        return len(self.ids)

    def __getitem__(self, item):
        '''
        根据item，返回图片和它对应的标注图片
        :param item: 框架指定，请勿修改
        :return: 字典{'img':FloatTensor类型,'mask'：FloatTensor类型}
        '''
        idx = self.ids[item]
        mask_file = glob(self.masks_dir + idx + '.*')  # 获取指定文件夹下文件名(列表)
        img_file = glob(self.imgs_dir + idx + '.*')

        img_path = img_file[0]
        mask_path = mask_file[0]

        assert len(img_file) == 1, \
            f'未找到图片 {idx}: {img_file}'

        assert len(mask_file) == 1, \
            f'未找到图片掩膜{idx}: {mask_file}'
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        assert img.size == mask.size, \
            f'图片与掩膜 {idx} 大小不一致,图片： {img.size} 掩膜： {mask.size}'

        mask = self._otsu_bin(mask)  # 数据问题 需要先做一次二值化
        mask = mask/255
        # img = self._channel_combination(img)
        img = self._preprocess(img)
        mask = self._preprocess(mask)
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
        pass
