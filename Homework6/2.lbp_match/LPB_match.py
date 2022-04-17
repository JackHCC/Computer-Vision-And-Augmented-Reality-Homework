import cv2
import numpy as np


def load_image(file1, file2):
    img1 = cv2.imread(file1, 1)
    img2 = cv2.imread(file2, 1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2


# generating lbp descriptor
def get_lbph(grayimg):
    rows, cols = grayimg.shape
    lbpmem = np.zeros((rows, cols), np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            sum = 0;
            if (grayimg[i, j] < grayimg[i - 1, j - 1]):
                sum = 128 + sum
            if (grayimg[i, j] < grayimg[i - 1, j]):
                sum = sum + 64
            if (grayimg[i, j] < grayimg[i - 1, j + 1]):
                sum = sum + 32
            if (grayimg[i, j] < grayimg[i, j + 1]):
                sum = sum + 16
            if (grayimg[i, j] < grayimg[i + 1, j + 1]):
                sum = sum + 8
            if (grayimg[i, j] < grayimg[i + 1, j]):
                sum = sum + 4
            if (grayimg[i, j] < grayimg[i + 1, j - 1]):
                sum = sum + 2
            if (grayimg[i, j] < grayimg[i, j - 1]):
                sum = sum + 1
            lbpmem[i, j] = sum

    lbph = cv2.calcHist([lbpmem], [0], None, [256], [0, 255])
    return lbph


def lbp_match(lbph1, lbph2):
    diff = 0
    add = 0
    for i in range(256):
        diff = diff + (lbph1[i] - lbph2[i]) * (lbph1[i] - lbph2[i])
        add = add + lbph1[i] * lbph1[i] + lbph2[i] * lbph2[i]
    rate = diff / add
    return rate


def main():
    file1 = "b1.tif"
    for i in range(2, 12):
        file2 = 'b' + str(i) + '.tif'
        img1, img2 = load_image(file1, file2)
        cv2.imshow(file2, img2)
        lbph1 = get_lbph(img1)
        lbph2 = get_lbph(img2)
        rate = lbp_match(lbph1, lbph2)
        print(file2 + str(rate))


if __name__ == "__main__":
    main()
