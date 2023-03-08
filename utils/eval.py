import math
#
import torch

from utils.dice_loss import dice_coeff

# from skimage.metrics import mean_squared_error
import os
import numpy as np

import cv2


# print('--------------------------注意---------------------------------\
# 二值化的评价应特别注意“影子现象”\
# 即Ground Truth容易受到其生成过程过程的影响，从而使得与其相关性较大得那组数据虚高\
# 比如，在Otsu二值图A得基础上人工擦除噪点、修补断裂后，得到标注图B。那么B得评价指标就会虚高\
# 应当使用交叉比对的方式，即图B与其它方法得到的二值图进行对比\
# -------------------------------------------------------------------------------')
def tensor_to_ndarray(t):
    cpu_tensor = t.cpu()
    res = cpu_tensor.detach().numpy()  # 转回numpy
    # print(res.shape)
    res = np.squeeze(res, 1)
    # res = np.swapaxes(res, 0, 2)
    # res = np.swapaxes(res, 0, 1)

    return res


def tp(original, contrast):
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
    if TP == 0:
        return 0
    recall = round(((TP) / (TP + FN)) * 100, 2)
    return recall


def Accuracy(TP, TN, FP, FN):
    if TP + TN == 0:
        return 0
    acc = (TP + TN) / (TP + TN + FP + FN) * 100
    return round(acc, 2)


def Precision(TP, FP):
    if TP == 0:
        return 0
    precision = ((TP) / (TP + FP)) * 100
    return round(precision, 2)


def F1measure(precision, recall):
    if precision == 0 or recall == 0:
        return 0.0
    f1measure = 2 * precision * recall / (precision + recall)
    return round(f1measure, 2)


def fm(pred, mask):
    pred = pred.cpu().detach().numpy()
    pred = (pred > 0.5).astype(np.int_)
    mask = mask.cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.int_)
    TP = tp(pred, mask)
    FP = fp(pred, mask)
    FN = fn(pred, mask)
    TN = tn(pred, mask)
    recall = round(Recall(TP, FN), 2)
    precision = round(Precision(TP, FP), 2)
    Fm = round(F1measure(precision, recall), 2)
    return Fm


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    print('PSNR:' + str(20 * math.log10(PIXEL_MAX / math.sqrt(mse))))
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# def mse(imageA, imageB):
#     err = mean_squared_error(imageA[:, :, 2], imageB[:, :, 2])
#     return err
def eval_fm(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_fm = 0.0
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred = net(imgs)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
            tot_fm += round(fm(pred, true_masks), 2)
    print(tot_fm)
    net.train()
    return tot / n_val, round(tot_fm / n_val, 2)


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32  # torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            imgs = imgs.transpose(3, 1).transpose(2, 3)
            mask_pred = net(imgs)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()

    net.train()
    return tot / n_val


def evaluation_metrics(pred_dir, gt_dir):
    pred_pth_list = os.listdir(pred_dir)
    gt_pth_list = os.listdir(gt_dir)
    print(pred_pth_list)
    assert len(pred_pth_list) == len(gt_pth_list), '列表长度异常'
    tot_f = 0.0
    tot_p = 0.0
    tot_r = 0.0
    for i in range(len(pred_pth_list)):
        print(f'------------{i}---------------')
        pred = cv2.imread(os.path.join(pred_dir, pred_pth_list[i]), 0)  # [200:1200,300:4100]
        _, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.imread(os.path.join(gt_dir, gt_pth_list[i]), 0)  # [200:1200,300:4100]
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        TP = tp(pred, mask)
        FP = fp(pred, mask)
        FN = fn(pred, mask)
        TN = tn(pred, mask)
        recall = Recall(TP, FN)
        precision = Precision(TP, FP)
        Fm = F1measure(precision, recall)
        tot_r += recall
        tot_p += precision
        tot_f += Fm
        print('fm:' + str(Fm))
        print('precision:' + str(precision))
        print('recall:' + str(recall))
        # iou(pred,mask)
        psnr(pred, mask)
        print('+++++++++++++++++++++++++++++++++++++++')
    avg_f = tot_f / len(pred_pth_list)
    avg_p = tot_p / len(pred_pth_list)
    avg_r = tot_r / len(pred_pth_list)
    print(f'f:{avg_f},p:{avg_p},r:{avg_r}')


# evaluation_metrics(r'D:\hongpu\data\val-rst',r'D:\hongpu\data\val_mask')

# valFileNameList = ['006.187', '011.225', '028.39', '035.127', '038.7', '059.301', '011.10', '017.26', '064.124',
#                        '075.106', '088.353', '092.160', '048.162', '042.402', '088.402', '046.252', '013.66', '027.345',
#                    '043.236', '071.265', '077.262', '044.237', '027.14', '001.3', '094.35']
# import os
#
# def temp():
#
#     for fn in valFileNameList:
#         sfn = os.path.join(r'E:\Dataset\oriImage\\', fn+r'.png')#source file name
#         dfn = os.path.join(r'E:\Dataset\temp\ori\\', fn+'.png')
#         command = fr'copy {sfn} {dfn}'
#         print(command)
#         os.system(command)
# temp()

