#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2

img_bgr = cv2.imread("./gem_flag.jpg", 1)

cv2.imshow("flag", img_bgr)

rows, cols, _ = img_bgr.shape

for r in range(rows):
    for c in range(cols):
        if img_bgr[r][c][0] < 60 and img_bgr[r][c][1] < 60 and img_bgr[r][c][2] < 60:
            img_bgr[r][c][0] += 200
            img_bgr[r][c][1] += 200
            img_bgr[r][c][2] += 200

        if img_bgr[r][c][0] < 140 and img_bgr[r][c][1] < 100 and img_bgr[r][c][2] > 140:
            img_bgr[r][c][0] = 255
            img_bgr[r][c][1] = 0
            img_bgr[r][c][2] = 0

        if 0 < img_bgr[r][c][0] < 100 and 100 < img_bgr[r][c][1] < 220 and img_bgr[r][c][2] > 130:
            img_bgr[r][c][0] = 0
            img_bgr[r][c][1] = 0
            img_bgr[r][c][2] = 255

cv2.imshow("replace_color", img_bgr)

cv2.waitKey(0)
