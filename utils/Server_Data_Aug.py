from os import listdir
from os.path import splitext
from tqdm import tqdm
import cv2


def get_files_name(dir_path):
    '''
    返回指定文件夹内的文件名列表
    :param dir_path: 问价夹路径
    :return:指定文件夹内的文件名列表
    '''
    ids = [splitext(file)[0] for file in listdir(dir_path)  # 获取图片名称，ids是一个列表
           if not file.startswith('.')]
    return ids


def presect(dir_path):
    """
    把dir_path下所有的图片进行切割
    :param dir_path:
    :return:
    """
    ids = get_files_name(dir_path)
    length = len(ids)
    for i in tqdm(range(0,length)):
        _overlapping_seg(fr'{dir_path}/{ids[i]}.png',f'{ids[i]}')
    print('done')


def _seg_img(img_path,img_name):
    """
    将任意大小的图片放缩至5376*1408。随后长轴分八份，纵轴分四份进行切割。
    :param img_path: 图片路径
    :param sect_path: 切分后图片存放路径
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, (5376, 1408))
    w = 672
    h = 352
    for i in range(0, 8):
        for j in range(0, 4):
            roi = img[j * h:(j + 1) * h, i * w:(i + 1) * w]
            retval = cv2.imwrite(f"../data/imgs/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"

def _smaller_seg_img(img_path,img_name):
    """
    将任意大小的图片放缩至5376*1408。随后长轴分17份，纵轴分7份进行切割。
    :param img_path: 图片路径
    :param sect_path: 切分后图片存放路径
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, (5440, 1344))
    w = 320
    h = 192
    for i in range(0, 17):
        for j in range(0, 7):
            roi = img[j * h:(j + 1) * h, i * w:(i + 1) * w]
            retval = cv2.imwrite(f"../data/mask_backup/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"

def _overlapping_seg(img_path,img_name):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param img_name: 待切图片名称
    :return:
    '''
    img = cv2.imread(img_path)
    h,w,c = img.shape

    patch_h = 256
    patch_w = 256
    stride_h = 128
    stride_w = 128

    n_w = int((w-patch_w)/stride_w)*stride_w+patch_w
    n_h = int((h-patch_h)/stride_h)*stride_h+patch_h

    img = cv2.resize(img, (n_w, n_h))
    n_patch_h = (h-patch_h)//stride_h+1
    n_patch_w = (w-patch_w)//stride_w+1
    n_patches = n_patch_h*n_patch_w

    for i in range(n_patch_w):
        for j in range(n_patch_h):
            y1 = j * stride_h
            y2 = y1 + patch_h
            x1 = i * stride_w
            x2 = x1 + patch_w
            roi = img[y1:y2,x1:x2]
            retval = cv2.imwrite(fr"/home/gpus_303/Ocean/unet-vanilla/data/mask/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"




def recurrennt_seg(img_path,img_name):
    '''
    循环切片
    :param img_path:
    :param img_name:
    :return:
    '''
    patch_size = 160
    img = cv2.imread(img_path)
    (h,w,c) = img.shape
    print(h, w, c)
    h_c = (int(h / patch_size) + 1)
    w_c = (int(w / patch_size) + 1)
    img = cv2.resize(img, (w_c*patch_size, h_c*patch_size))
    for i in range(0, w_c):
        for j in range(0, h_c):
            roi = img[j * patch_size:(j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
            retval = cv2.imwrite(f"E:/Project/Unet-vanilla/data/temp-mask/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"


def main():
    presect(r'/home/gpus_303/Ocean/unet-vanilla/raw_data/a_mask_dir')

main()