import cv2
import numpy as np


def main():
    img = cv2.imread("cboard.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    outImage = cv2.drawKeypoints(img, kp, (255, 0, 0))

    cv2.imshow("sift", outImage)
    cv2.imwrite("siftpoint.jpg", outImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
