"""
Sample for testure synthesis
Course of CV & AR,SSPKU
Lin Jinlong, March 8,2020
"""
import numpy as np
from random import randint
from time import time
import cv2

import Horizontal as hor
import Vertical as ver
import SearchForSimilarAreas as sea

# define patch size
patchsize = 16
overlappingsize = 8;
# read file 
texturepic = cv2.imread("basket.jpg", 1)
pic = cv2.imread("girl.jpg", 1)
# showing
cv2.imshow("picture", pic)
cv2.imshow("texturepic", texturepic)

prows, pcols, ch = pic.shape
trows, tcols, ch = texturepic.shape

print(prows, pcols, trows, tcols)
# stride size
stride = patchsize - overlappingsize
rowspos = 0
colspos = 0

# allocating space
transferpic = np.zeros((prows, pcols, 3), dtype=np.uint8)
trowsf = patchsize
tcolsf = pcols
for w in range(0, prows, stride):
    rangerow = patchsize
    if (w + patchsize > prows):
        rangerow = prows - w

    # save a horizontal strip
    colspatch = np.zeros((patchsize, pcols, 3), dtype=np.uint8)
    # the filled block size
    colsf = patchsize
    rowsf = rangerow
    for i in range(0, pcols, stride):
        rangecol = patchsize
        if (i + patchsize > pcols):
            rangecol = pcols - i

        # get patch from input image
        patch = pic[w:w + rangerow, i:i + rangecol]

        # finding the Corresponding area in texturpic
        # cv2.imshow("patch",patch)
        similarpatch = sea.SearchForSimilarAreas(patch, texturepic)
        # cv2.imshow("spatch",similarpatch)
        if (i == 0):
            colspatch[0:rowsf, 0:colsf] = similarpatch[0:rowsf, 0:colsf]
        else:
            # merge the horizontal
            if (rangecol > overlappingsize):
                incolspatch = colspatch[0:rowsf, 0:colsf]
                colstemppatch = hor.Horizontal(incolspatch, similarpatch, overlappingsize)
            else:
                break
            rowsf, colsf, ch = colstemppatch.shape  # get current shape
            if (rowsf > rangerow):  # for bottom edge
                colspatch[0:rangerow, 0:colsf] = colstemppatch[0:rangerow, 0:colsf]
            else:
                colspatch[0:rowsf, 0:colsf] = colstemppatch[0:rowsf, 0:colsf]

        # cv2.imshow("colspatch",colspatch)
    if (w == 0):
        transferpic[0:trowsf, 0:tcolsf] = colspatch
    else:
        if (rangerow > overlappingsize):
            # merge the Vertical
            intransferpic = transferpic[0:trowsf, 0:tcolsf]
            transfertemppic = ver.Vertical(intransferpic, colspatch, overlappingsize)
            trowsf, tcolsf, ch = transfertemppic.shape  # retcurrent shape
        else:
            break;
        print(trowsf)
        # if(trowsf>prows):  #for bottom edge
        #     trowsf=prows
        if (trowsf < prows):
            transferpic[0:trowsf, 0:tcolsf] = transfertemppic
        else:
            transferpic[0:prows, 0:tcolsf] = transfertemppic[0:prows, 0:tcolsf]
        # cv2.imshow("pic",transfertemppic)

cv2.imshow("transferpic", transferpic)

cv2.imwrite("transfergirl.jpg", transferpic)

while 1:
    key = cv2.waitKey(1)
    if key > 0:
        break
cv2.destroyAllWindows()
