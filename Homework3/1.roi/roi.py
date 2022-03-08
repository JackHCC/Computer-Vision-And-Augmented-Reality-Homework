#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Homework 
@File    :roi.py
@Author  :JackHCC
@Date    :2022/3/8 9:10 
@Desc    :

'''
import cv2
import numpy as np
import time

# read image
inimage = cv2.imread("finger.jpg", 1)

cv2.imshow("finger.jpg", inimage)
print(inimage.shape)

# allocating image space color  ROI [ymin:ymax, xmin:xmax]
# 767*403
ROIimg = inimage[0:403, 20:787]  # for cinema
cv2.imwrite("finger_ROI.jpg", ROIimg)
col, row, ch = ROIimg.shape
print(col, row, ch)

# set the source point in source image
inpoints = np.float32([(0, 0), (767, 0), (50, 403), (737, 403)])  # p1,p2,p3,p4, 可以在ROI中取点，也可以直接从原图中取点;变换时输入提取参考点数组的图像。

# inpoints=np.float32([(180,71),(341,49),(183,146),(338,166)])#p1,p2,p3,p4

outpoints = np.float32([(0, 0), (639, 0), (0, 639), (639, 639)])  # p'1,p'2,p'3, p'4

# get Transform parameters
Trans = cv2.getPerspectiveTransform(np.array(inpoints), np.array(outpoints))

# Transform
start = time.time()
transimg = cv2.warpPerspective(ROIimg, Trans, (640, 640))  # 输出图像可以改变，但变换关系不变
end = time.time()
print("time=", end - start)

cv2.imshow('ROI', ROIimg)
cv2.imwrite("finger_out.jpg", transimg)

cv2.imshow('output', transimg)

while 1:
    key = cv2.waitKey(1)

    if key > 0:
        break

cv2.destroyAllWindows()
