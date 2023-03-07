
import cv2
import numpy as np
import random
from tqdm import tqdm
def draw_rect(img):
    x1 = random.randint(0,250)
    y1 = random.randint(0,250)
    x2 = random.randint(0,250)
    y2 = random.randint(0,250)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1)
    return img

def draw_circle(img,mask):
    center = random.randint(350,450), random.randint(100,150)
    r = random.randint(20,50)
    cv2.circle(img, (center), r, (255, 0, 0), -1)
    cv2.circle(mask, (center), r, (255, 255, 255), -1)
    return img,mask
def draw_poly(img):
    rect = np.array([[[random.randint(0,150), random.randint(250,500)], [random.randint(151,300), random.randint(250,500)], [random.randint(301,500), random.randint(250,500)]]], np.int32)
    cv2.fillPoly(img,rect,(0,0,255))
    return img

for i in tqdm(range(50)):

    img = np.zeros((500, 500, 3), np.uint8)
    mask = np.zeros((500, 500, 3), np.uint8)
    img = draw_rect(img)
    img,mask = draw_circle(img,mask)
    img = draw_poly(img)
    cv2.imwrite(fr'../data/temp/{str(i)}.png',img)
    # cv2.imwrite(fr'../data/test_mask/{str(i)}.png', mask)
    # cv2.imshow('rectangle',img)
    # cv2.waitKey()