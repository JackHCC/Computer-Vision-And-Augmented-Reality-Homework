"""
Sample for testure synthesis
Course of CV & AR,SSPKU
Lin Jinlong, March 8,2020
"""
import numpy as np
from random import randint
from time import time
import cv2
import struct


# define the comparation function

def compare_minimumDifferenceLocationArray(arrayst1, arrayst2):
    row_num1, col_num1, pixelDifference1 = struct.unpack('iii', arrayst1)
    row_num2, col_num2, pixelDifference2 = struct.unpack('iii', arrayst2)
    if (pixelDifference1 > pixelDifference2):
        return 1
    else:
        return 0


def SearchForSimilarAreas(currentpatch, texturepic):
    INT_MAX = 2147483647
    speedupValue = 3
    arrayst1 = struct.pack('iii', 0, 0, INT_MAX)
    # getting image size
    patchrows, patchcols, ch = currentpatch.shape
    texturerows, texturecols, ch = texturepic.shape

    # the max number of pixel rows and cols
    rows_need_be_compared = texturerows - patchrows
    cols_need_be_compared = texturecols - patchcols

    # transfer the image to gray
    graycurrentpatch = cv2.cvtColor(currentpatch, cv2.COLOR_BGR2GRAY, 0)
    graytexturepic = cv2.cvtColor(texturepic, cv2.COLOR_BGR2GRAY, 0)

    # comparing the patch and then finding the best suitable one

    for i in range(0, rows_need_be_compared, speedupValue):
        for j in range(0, cols_need_be_compared, speedupValue):
            graytexturepatch = graytexturepic[i:i + patchrows, j:j + patchcols]
            gray_difference_sum = np.sum(abs(graycurrentpatch - graytexturepatch))
            # print(gray_difference_sum)
            arrayst2 = struct.pack('iii', i, j, gray_difference_sum)
            if (compare_minimumDifferenceLocationArray(arrayst1, arrayst2) == 1):
                # print("OK")
                arrayst1 = arrayst2

    # get the texture patch
    minrows, mincols, mimsum = struct.unpack('iii', arrayst1)
    # print(minrows,mincols,mimsum)

    texturepatch = texturepic[minrows:minrows + patchrows, mincols:mincols + patchcols]

    return texturepatch
