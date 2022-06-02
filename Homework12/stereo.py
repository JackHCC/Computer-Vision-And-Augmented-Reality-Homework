import cv2
import numpy as np


imgL = cv2.imread('im0.png')
imgR = cv2.imread('im1.png')

imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=7)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=7)
disp = stereo.compute(imgLG, imgRG).astype(np.float32) / 16.0

cv2.namedWindow('BM disparity', cv2.WINDOW_FREERATIO)
cv2.imshow('BM disparity', (disp - 0) / 128)
cv2.waitKey(0)

# window_size = 7
# min_disp = 32  # 最小视差数
# num_disp = 288 - min_disp
# blockSize = window_size
# uniquenessRatio = 1  # 最佳比配代价优于次佳代价的比例
# speckleRange = 2  # 相连物体最大视差变化 *16 pixels
# speckleWindowSize = 3
# disp12MaxDiff = 200  # 左右最大允许视差  pixels
# P1 = 600  # 视差平滑控制参数1
# P2 = 2400  # 视差平滑控制参数2
#
# stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
#                                uniquenessRatio=uniquenessRatio, speckleRange=speckleRange,
#                                speckleWindowSize=speckleWindowSize, disp12MaxDiff=disp12MaxDiff, P1=P1, P2=P2)

# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 16
num_disp = 192 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=7,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32
                               )

disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
cv2.namedWindow('SGBM disparity', cv2.WINDOW_FREERATIO)
cv2.imshow('SGBM disparity', (disp - min_disp) / num_disp)
cv2.waitKey(0)
