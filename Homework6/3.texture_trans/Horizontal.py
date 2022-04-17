"""
Sample for testure synthesis
Course of CV & AR,SSPKU
Lin Jinlong, March 8,2020
"""
import numpy as np
from random import randint
from time import time
import cv2


def Horizontal(rock1_, rock2_, cutCols):  # rock1_ ：inimage, rock2_：texturepatch

    DBL_MAX = 2147483647
    rows1, cols1, ch = rock1_.shape
    rows2, cols2, ch = rock2_.shape

    # print(rows1, cols2)
    # get the leftpart and the right part
    leftpart = rock1_[0:rows1, cols1 - cutCols:cols1]
    rightpart = rock2_[0:rows2, 0:cutCols]

    # calculate the difference matrix
    differ_matrix = np.zeros((cutCols, rows1), dtype=np.int)
    # computing
    for i in range(0, cutCols):
        for j in range(0, rows1):
            # d=abs(R1-R2)+abs(B1-B2)+abs（G1-G2）
            differ_matrix[i][j] = abs(int(leftpart[j, i, 0]) - int(rightpart[j, i, 0])) + abs(
                int(leftpart[j, i, 1]) - int(rightpart[j, i, 1])) + abs(
                int(leftpart[j, i, 2]) - int(rightpart[j, i, 2]))
            # print(differ_matrix[i][j])
    # searching the best joints set
    f = np.zeros((cutCols, rows1), np.int32)  # record the shortest distace
    l = np.zeros((cutCols, rows1), np.int32)  # record assenmbly path

    # initialize first step as itself
    for i in range(0, cutCols):
        l[i][0] = i

    # intialize the pixel value of the first col
    for i in range(0, cutCols):
        f[i][0] = differ_matrix[i][0]

    # searching
    for i in range(0, rows1):
        for j in range(0, cutCols):
            # if there are three path can be choosed
            if (j != 0 and j != (cutCols - 1)):
                # if middle path is shortest
                if (((f[j][i - 1] + differ_matrix[j][i]) < (f[j - 1][i - 1] + differ_matrix[j][i])) and (
                        (f[j][i - 1] + differ_matrix[j][i]) < (f[j + 1][i - 1] + differ_matrix[j][i]))):
                    f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                    l[j][i] = j

                # if upper path is shortest
                elif (((f[j - 1][i - 1] + differ_matrix[j][i]) < (f[j][i - 1] + differ_matrix[j][i])) and (
                        (f[j - 1][i - 1] + differ_matrix[j][i]) < (f[j + 1][i - 1] + differ_matrix[j][i]))):
                    f[j][i] = f[j - 1][i - 1] + differ_matrix[j][i]
                    l[j][i] = j - 1

                # if lower path is shortest
                elif (((f[j + 1][i - 1] + differ_matrix[j][i]) < (f[j][i - 1] + differ_matrix[j][i])) and (
                        (f[j + 1][i - 1] + differ_matrix[j][i]) < (f[j - 1][i - 1] + differ_matrix[j][i]))):
                    f[j][i] = f[j + 1][i - 1] + differ_matrix[j][i]
                    l[j][i] = j + 1
                # if there are two path distance are equal, choose middle path by default.  I am a lazy man...
                else:
                    f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                    l[j][i] = j

            # if there are two path can be choosed, the top line or the bottom line
            else:
                # the top line
                if (j == 0 and cutCols > 1):
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
                elif (j == (cutCols - 1) and cutCols > 1):
                    # if middle path is shortest
                    if (f[j][i - 1] + differ_matrix[j][i] < f[j - 1][i - 1] + differ_matrix[j][i]):
                        f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                        l[j][i] = j
                        # if upper path is shortest
                    elif ((f[j - 1][i - 1] + differ_matrix[j][i]) < (f[j][i - 1] + differ_matrix[j][i])):
                        f[j][i] = f[j - 1][i - 1] + differ_matrix[j][i]
                        l[j][i] = j - 1
                    # if middle and upper path are euqal, choose the middle path by default
                    else:
                        f[j][i] = f[j][i - 1] + differ_matrix[j][i];
                        l[j][i] = j

                # if there is only one path can be choose
                elif (cutCols == 1):
                    f[j][i] = f[j][i - 1] + differ_matrix[j][i]
                    l[j][i] = j
                else:
                    exit(-1)

    minDistance = DBL_MAX
    position = -1
    lineNum = -1
    # record the shortest distance path

    boundaryPosition = np.zeros(rows1, np.int32)
    # find the shortest distance from left boundary to right boundary
    for i in range(0, cutCols):
        if (f[i][rows1 - 1] < minDistance):
            minDistance = f[i][rows1 - 1]
            position = i
        else:
            continue

    # starting from the shortest distance path last pixel
    lineNum = l[position][rows1 - 1]
    for i in range(rows1 - 1, 0, -1):
        lineNum = l[lineNum][i]  # from which row to here, upper, middle or lower
        boundaryPosition[i] = lineNum  # record the shortest distance path each step
        # print(lineNum)

    # merge patch
    vmergepatch = np.zeros((rows1, cols1 + cols2 - cutCols, 3), np.uint8)
    for i in range(0, rows1):
        for j in range(0, cols2 + cols1 - cutCols):
            if (j <= boundaryPosition[i] + (cols1 - cutCols)):
                vmergepatch[i][j] = rock1_[i][j]
            else:
                vmergepatch[i][j] = rock2_[i][j - (cols1 - cutCols)]

    return vmergepatch
