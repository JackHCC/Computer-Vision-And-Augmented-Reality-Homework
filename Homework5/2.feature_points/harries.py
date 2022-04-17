import cv2
import numpy as np

filename = 'cboard.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("crossbar", gray)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)

while 1:
    key = cv2.waitKey(0)
    if key > 0:
        break
cv2.destroyAllWindows()
