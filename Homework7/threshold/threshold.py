import cv2

inputImgfile = 'finger1.tif'

# feature number
featureSum = 0
inimg = cv2.imread(inputImgfile, 0)

img = cv2.GaussianBlur(inimg, (27, 27), 25, 25)

cv2.imshow("blur", img)

# ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)

# print(ret)
cv2.imshow('th', th1)
while 1:
    key = cv2.waitKey(1)
    if key > 0:
        break
cv2.destroyAllWindows()
