import numpy as np
import cv2


def main():
    # img = cv2.imread('plane.jpg')
    img=cv2.imread('./girl.jpg')

    cv2.imshow("plane", img)
    row, col, channel = img.shape
    print("row=", row)
    print("col=", col)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect = (100, 50, 220, 400)  # the interesting area
    rect = (100, 50, 400, 300)  # the interesting area

    imgrect = img.copy()
    # imgrect = cv2.rectangle(imgrect, (50, 100), (400, 200), (0, 0, 255), 2)
    imgrect = cv2.rectangle(imgrect, (50, 100), (300, 400), (0, 0, 255), 2)

    cv2.imshow("rect", imgrect)

    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # print(fgdModel)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    cv2.imshow('plane', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()