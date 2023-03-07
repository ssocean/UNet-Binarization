import os
import random
from os import listdir
from os.path import splitext
from tqdm import tqdm
import numpy as np
import cv2
from workshop.GeneralTools import *


def showim(img):
    '''
    展示图片
    :param img: ndarray格式的图片
    '''
    cv2.namedWindow(f'img', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f'img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stain(img, dirt_num=3):
    """
    在图片上模拟圆斑污损
    :param img: 输入np图像
    :param dirt_num: 圆斑数目 默认3
    :return: 已进行模拟污损的np图片
    """
    imshape = img.shape
    h = imshape[0]  # 192
    w = imshape[1]
    assert h / 10 < w / 3, f'h/10应当小于h/3，然而{h / 10}>{w / 3}'
    masks = np.zeros((h, w, 3), np.uint8)
    # 随机绘制实心圆
    for i in range(0, dirt_num + 1):
        # 随机中心点
        center_x = np.random.randint(0, high=w)
        center_y = np.random.randint(0, high=h)

        # 随机半径与颜色
        radius = np.random.randint(low=h / 10, high=w / 3)
        r = random.randint(140, 160)
        g = random.randint(110, 130)
        b = random.randint(60, 90)
        color = [b, g, r]
        cv2.circle(masks, (center_x, center_y), radius, color, -1)

    dst = blend(img, masks, r_threshold=140)
    return dst


def dot(img, dot_num=100):
    """
    在图片上模拟斑点污损
    :param img: 输入np图像
    :param dot_num: 圆斑数目 默认100
    :return: 已进行模拟污损的np图片
    """
    imshape = img.shape

    h = imshape[0]
    w = imshape[1]
    masks = np.zeros((h, w, 3), np.uint8)
    # 2.循环随机绘制实心圆
    for i in range(0, dot_num + 1):
        # 随机中心点
        center_x = np.random.randint(0, high=w)
        center_y = np.random.randint(0, high=h)

        # 随机半径与颜色
        radius = np.random.randint(1, high=h / 30)
        r = random.randint(150, 160)
        g = random.randint(140, 150)
        b = random.randint(110, 120)
        color = [b, g, r]
        cv2.circle(masks, (center_x, center_y), radius, color, -1)

    dst = blend(img, masks, r_threshold=150)
    return dst


def scratch(img):
    """
    在图片上模拟划痕污损
    :param img:输入np图像
    :return:已进行模拟污损的np图片
    """
    imshape = img.shape
    # print(imshape[0])
    h = imshape[0]
    w = imshape[1]

    masks = np.zeros((h, w, 3), np.uint8)
    r = random.randint(140, 150)
    g = random.randint(130, 140)
    b = random.randint(100, 110)
    color = [b, g, r]

    l = random.randint(1, 5)
    left_top_x = random.randint(1, int(w / l))
    left_top_y = random.randint(1, int(h / l))
    right_botom_x = random.randint(int(w / (l + 1)), int(w / (l)))
    right_botom_y = random.randint(int(h / (l + 1)), int(h / (l)))
    cv2.line(masks, (left_top_x, left_top_y), (right_botom_x, right_botom_y), color, 3)

    dst = blend(img, masks, r_threshold=140)
    return dst


def blend(img, mask, r_threshold, alpha=0.5, is_vis=False):
    """
    模拟一张污渍图，需要原图与生成的
    :param img: 原图
    :param mask: 污渍图
    :param r_threshold:污渍图R信道阈值
    :param alpha: 混合比例
    :param is_vis: 是否显示结果
    :return: 混合后的np图片
    """
    imshape = img.shape
    res = np.zeros(imshape, dtype=np.uint8)
    beta = 1 - alpha
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            if mask[i][j][2] >= r_threshold:
                res[i][j] = alpha * img[i][j] + beta * mask[i][j]
            else:
                res[i][j] = img[i][j]
    if is_vis:
        showim(res)
    return res


def flip_y(img):
    rows, cols = img.shape[:2]
    mapx = np.zeros(img.shape[:2], dtype=np.float32)
    mapy = np.zeros(img.shape[:2], dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            mapx.itemset((i, j), j)
            mapy.itemset((i, j), rows - 1 - i)
    rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return rst


def flip_x(img):
    rows, cols = img.shape[:2]
    mapx = np.zeros(img.shape[:2], dtype=np.float32)
    mapy = np.zeros(img.shape[:2], dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            mapx.itemset((i, j), cols - 1 - j)
            mapy.itemset((i, j), i)
    rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return rst


def flip_xy(img):
    rows, cols = img.shape[:2]
    mapx = np.zeros(img.shape[:2], dtype=np.float32)
    mapy = np.zeros(img.shape[:2], dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            mapx.itemset((i, j), cols - 1 - j)
            mapy.itemset((i, j), rows - 1 - i)
    rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return rst


def flip(img, mask):
    method = random.randint(0, 3)
    method = 2
    if method == 0:
        img = flip_x(img)
        mask = flip_x(mask)
    elif method == 1:
        img = flip_y(img)
        mask = flip_y(mask)
    elif method == 2:
        img = flip_xy(img)
        mask = flip_xy(mask)
    return img, mask


def background(h, w):
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = random.randint(100, 120)
    background[:, :, 1] = random.randint(140, 160)
    background[:, :, 2] = random.randint(170, 190)
    res = dot(background, dot_num=10000)
    return res


def get_files_name(dir_path):
    '''
    返回指定文件夹内的文件名列表
    :param dir_path: 问价夹路径
    :return:
    '''
    ids = [splitext(file)[0] for file in listdir(dir_path)  # 获取图片名称，ids是一个列表
           if not file.startswith('.')]
    return ids


def img_tf(dir_path, dst_path):
    ids = get_files_name(dir_path)
    length = len(ids)
    for i in tqdm(range(0, length)):
        img = cv2.imread(fr'{dir_path}/{ids[i]}.png')
        rst = stain(img)  # 更改此函数以更换模拟效果
        ret = cv2.imwrite(fr'{dst_path}/{ids[i]}_stain.png', rst)  # 更改路径
        assert ret, 'save failed'
        # print(i)
    print('done')


def img_tf_flip(img_path, mask_path):
    img_ids = get_files_name(img_path)
    mask_ids = get_files_name(mask_path)
    length = len(img_ids)
    for i in tqdm(range(0, length)):
        img = cv2.imread(fr'{img_path}/{img_ids[i]}.png')
        mask = cv2.imread(fr'{mask_path}/{mask_ids[i]}.png')

        img, mask = flip(img, mask)
        ret = cv2.imwrite(fr'../data/img/{img_ids[i]}_flip2.png', img)
        assert ret, 'save failed'
        ret = cv2.imwrite(fr'../data/mask/{mask_ids[i]}_flip2.png', mask)
        assert ret, 'save failed'
        # print(i)
    print('done')
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





def test():
    pass


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

# img_tf(r'../dataset/origin_img_patch',r'../dataset/tf_img_stain')

# dir_bit_or(r'C:\Users\OCEAN\Desktop\prediction_result\2_gpu_slice', r'C:\Users\OCEAN\Desktop\prediction_result\1_cpu_whole') #第一次
# dir_bit_or(r'C:\Users\OCEAN\Desktop\prediction_result\8-gpu-resize', r'C:\Users\OCEAN\Desktop\prediction_result\7-cpu') #第二次

# dir_bit_or(r'C:\Users\OCEAN\Desktop\res_R2Att_gpu', r'G:\Dataset\Unet_bin_img',r'C:\Users\OCEAN\Desktop\res_R2Att_bitwise') #R2attunet bitwise res
# img_bit_or()

# copy_mask()
# from skimage import data, io, util #'0.16.2'
# import matplotlib.pyplot as plt #'3.0.3'
import PIL #'6.2.1'
import cv2 #'4.1.1'
# check = util.img_as_bool(data.checkerboard())
def patchit(root_dir,dst_dir):
    auto_make_directory(dst_dir)
    file_paths = get_files_pth(root_dir)
    for pth in file_paths:
        img = cv2.imread(pth,0)#[300:1100,500:4800]
        # img = cv2.resize(img,(5400,1497))[300:1100,500:4800]
        auto_make_directory(dst_dir)
        _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(dst_dir,os.path.basename(pth)),img, [cv2.IMWRITE_PNG_BILEVEL, 1])
        # img = cv2.resize(img, (1000, 800))
        # cv2.imwrite(os.path.join(dst_dir, os.path.basename(pth)), img)
patchit(r'G:\Paper\Bin_result\collection\resize_dibco\5-21-att',r'G:\Paper\Bin_result\collection\patchit_dibco')
# def change2png(dir):
#     file_paths = get_files_pth(dir)
#     file_name = get_files_name(dir)
#     for i,pth in enumerate(file_paths):
#         img = cv2.imread(pth,0)#[300:1100,500:4800]
#         cv2.imwrite(r'G:\Dataset\DIBCO2017_GT\GT-PNG/'+file_name[i]+'.png',img)
# change2png('G:\Dataset\DIBCO2017_GT\GT')