"""
Sample for testure synthesis
Course of CV & AR,SSPKU
Lin Jinlong, March 8,2020
"""
import numpy as np
from random import randint
from time import time
import cv2


def Vertical(rock1_, rock2_, cutRows):
    DBL_MAX = 2147483647
    rows1, cols1, ch = rock1_.shape
    rows2, cols2, ch = rock2_.shape
    # print(rows1,cols1,rows2,cols2)
    rows = rows2
    cols = cols1
    # print(rows,cols)
    # get uppart
    uppart = rock1_[rows - cutRows:rows, 0:cols]
    downpart = rock2_[0:cutRows, 0:cols]

    # calculate the difference matrix
    differ_matrix = np.zeros((cutRows, cols), dtype=np.int32)
    # x,y=differ_matrix.shape
    # print(x,y)
    for i in range(0, cutRows):
        for j in range(0, cols):
            # print(i,j)
            # d=abs(R1-R2)+abs(B1-B2)+abs（G1-G2）
            differ_matrix[i][j] = abs(int(uppart[i, j, 0]) - int(downpart[i, j, 0])) + abs(
                int(uppart[i, j, 1]) - int(downpart[i, j, 1])) + abs(int(uppart[i, j, 2]) - int(downpart[i, j, 2]))

    # searching the best joints set
    f = np.zeros((cutRows, cols), np.int32)  # record the shortest distace
    l = np.zeros((cutRows, cols), np.int32)  # record assenbly  path

    # initialize first stwp as itself
    for i in range(0, cutRows):
        l[i][0] = i

    # intialize the pixel value of the first col
    for i in range(0, cutRows):
        f[i][0] = differ_matrix[i][0]

    # searching
    for i in range(0, cols):
        for j in range(0, cutRows):
            # if there are three path can be choosed
            if (j != 0 and j != (cutRows - 1)):
                # if middle path is shortest
                if ((f[j][i - 1] + differ_matrix[j][i] < f[j - 1][i - 1] + differ_matrix[j][i]) and (
                        f[j][i - 1] + differ_matrix[j][i] < f[j + 1][i - 1] + differ_matrix[j][i])):
                    f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                    l[j][i] = j

                # if upper path is shortest
                elif ((f[j - 1][i - 1] + differ_matrix[j][i] < f[j][i - 1] + differ_matrix[j][i]) and (
                        f[j - 1][i - 1] + differ_matrix[j][i] < f[j + 1][i - 1] + differ_matrix[j][i])):
                    f[j][i] = f[j - 1][i - 1] + differ_matrix[j][i]
                    l[j][i] = j - 1

                # if lower path is shortest
                elif ((f[j + 1][i - 1] + differ_matrix[j][i] < f[j][i - 1] + differ_matrix[j][i]) and (
                        f[j + 1][i - 1] + differ_matrix[j][i] < f[j - 1][i - 1] + differ_matrix[j][i])):
                    f[j][i] = f[j + 1][i - 1] + differ_matrix[j][i]
                    l[j][i] = j + 1
                # if there are two path distance are equal, choose middle path by default.  I am a lazy man...
                else:
                    f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                    l[j][i] = j

            # if there are two path can be choosed, the top line or the bottom line
            else:
                # the top line
                if (j == 0 and cutRows > 1):
                    # if middle path is shortest
                    if (f[j][i - 1] + differ_matrix[j][i] < f[j + 1][i - 1] + differ_matrix[j][i]):
                        f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                        l[j][i] = j
                    # if lower path is shorest
                    elif (f[j + 1][i - 1] + differ_matrix[j][i] < f[j][i - 1] + differ_matrix[j][i]):
                        f[j][i] = f[j + 1][i - 1] + differ_matrix[j][i]
                        l[j][i] = j + 1
                    # if the middle and lower path are equal ,choose the middle path by default
                    else:
                        f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                        l[j][i] = j
                # the bottom line
                elif (j == (cutRows - 1) and cutRows > 1):
                    # if middle path is shortest
                    if (f[j][i - 1] + differ_matrix[j][i] < f[j - 1][i - 1] + differ_matrix[j][i]):
                        f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                        l[j][i] = j
                        # if upper path is shortest
                    elif (f[j - 1][i - 1] + differ_matrix[j][i] < f[j][i - 1] + differ_matrix[j][i]):
                        f[j][i] = f[j - 1][i - 1] + differ_matrix[j][i]
                        l[j][i] = j - 1
                    # if middle and upper path are euqal, choose the middle path by default
                    else:
                        f[j][i] = f[j][i - 1] + differ_matrix[j][i];
                        l[j][i] = j

                # if there is only one path can be choose
                elif (cutRows == 1):
                    f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                    l[j][i] = j
                else:
                    exit(-1)

    minDistance = DBL_MAX
    position = -1
    lineNum = -1
    # record the shortest distance path

    boundaryPosition = np.zeros(cols, dtype=np.int)
    # find the shortest distance from left boundary to right boundary
    for i in range(0, cutRows):
        if (f[i][cols - 1] < minDistance):
            minDistance = f[i][cols - 1]
            position = i
        else:
            continue

    # starting from the shortest distance path last pixel
    lineNum = l[position][cols - 1]
    for i in range(cols - 1, 0, -1):
        lineNum = l[lineNum][i]  # from which row to here, upper, middle or lower
        boundaryPosition[i] = lineNum  # record the shortest distance path each step

        # merge patch
    vmergepatch = np.zeros((rows1 + rows2 - cutRows, cols, 3), dtype=np.uint8)
    for i in range(0, rows1 + rows2 - cutRows):
        for j in range(0, cols):
            if (i <= boundaryPosition[j] + (rows1 - cutRows)):
                vmergepatch[i][j] = rock1_[i][j]
            else:
                vmergepatch[i][j] = rock2_[i - (rows1 - cutRows)][j]

    return vmergepatch
