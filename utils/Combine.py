import os
import numpy as np
import cv2

from workshop.GeneralTools import *


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

            rst = cv2.threshold()
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

def combine_p2p_rst(dir,rst_pth):
    i=0
    j=0
    patch_size = 512
    pred_np = np.zeros((512*3, 512*11), dtype=np.uint8)
    for img_pth in get_files_pth(dir):
        roi = cv2.imread(img_pth,0)
        roi = cv2.resize(roi,(patch_size,patch_size))
        fname = get_filename_from_pth(img_pth)
        orders = fname.split('_')
        i = int(orders[1])#column
        j = int(orders[2].split('.')[0])#row
        print(i,j)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred_np[j * patch_size:(j + 1) * patch_size,
            i * patch_size:(i + 1) * patch_size] = roi

    cv2.imwrite(rst_pth,pred_np)
    pass
# scale = str(2.8)
# pth = rf'G:\temp\resize_test\{scale}'
# combine_p2p_rst(pth,rf'G:\Paper\Bin_result\collection\resize_p2p\{scale}.png')
def showim(img:np.ndarray):
    '''
    展示图片
    :param img: ndarray格式的图片
    '''
    cv2.namedWindow(f'image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f'image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def dibco_combine(dir):
    '''
    将多个dibco文件取一部分合并
    :param dir: dibco结果
    :return:
    '''
    roi1 = cv2.imread(os.path.join(dir,'1.bmp'))[0:300,400:700]
    roi3 = cv2.imread(os.path.join(dir,'3.bmp'))[180:480,560:860]
    roi4 = cv2.imread(os.path.join(dir, '4.bmp'))[600:900, 1300:1600]
    roi5 = cv2.imread(os.path.join(dir, '5.bmp'))[200:500, 400:700]
    roi6 = cv2.imread(os.path.join(dir, '10.bmp'))[600:900, 1100:1400]
    roi7 = cv2.imread(os.path.join(dir, '7.bmp'))[0:300, 0:300]
    roi9 = cv2.imread(os.path.join(dir, '9.bmp'))[400:700, 600:900]
    roi10 = cv2.imread(os.path.join(dir, '10.bmp'))[200:500, 200:500]
    pred_np = np.zeros((600, 1200,3), dtype=np.uint8)

    pred_np[0:300,0:300] = roi1
    pred_np[0:300, 300:600] = roi3
    pred_np[0:300, 600:900] = roi4
    pred_np[0:300, 900:1200] = roi5
    pred_np[300:600, 0:300] = roi6
    pred_np[300:600, 300:600] = roi7
    pred_np[300:600, 600:900] = roi9
    pred_np[300:600, 900:1200] = roi10

    showim(pred_np)
    rst_path = r''
    cv2.imwrite(rst_path,pred_np)
dibco_combine(r'G:\Dataset\DIBCO2017_GT\Dataset')
