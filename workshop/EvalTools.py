import math

import numpy as np
from skimage.metrics import mean_squared_error


def tp(original, contrast):
    '''
    求出真阳值
    :param original: ndarray类型数据
    :param contrast: ndarray类型数据
    :return:
    '''
    return np.sum(np.multiply((original == 0).astype(int), \
                              (contrast == 0).astype(int)))


def fp(original, contrast):
    return np.sum(np.multiply((original == 0).astype(int), \
                              (contrast != 0).astype(int)))


def fn(original, contrast):
    return np.sum(np.multiply((original != 0).astype(int), \
                              (contrast == 0).astype(int)))


def tn(original, contrast):
    return np.sum(np.multiply((original != 0).astype(int), \
                              (contrast != 0).astype(int)))


# 定义 Recall, Precision
def Recall(TP, FN):
    recall = ((TP) / (TP + FN)) * 100
    return recall


def Accuracy(TP, TN, FP, FN):
    acc = (TP + TN) / (TP + TN + FP + FN) * 100
    return acc


def Precision(TP, FP):
    precision = ((TP) / (TP + FP)) * 100
    return precision


def F1measure(precision, recall):
    f1measure = 2 * precision * recall / (precision + recall)
    return f1measure



def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(imageA, imageB):
    err = mean_squared_error(imageA[:, :, 2], imageB[:, :, 2])
    return err

