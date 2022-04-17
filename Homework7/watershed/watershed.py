import numpy as np
import cv2


def genColor(value):
    a = ((value + 10 * value) * value) & 255
    b = ((value + 20 * value) * value) % 255;
    c = ((value + 30 * value) * value) % 255
    return (a, b, c)


def main():
    image = cv2.imread("town.jpg",1)
    # image = cv2.imread('../../flower.jpg', 1)
    #  cv2.imshow("town",image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 2)
    #  cv2.imshow("gray img",img_blur)
    img_canny = cv2.Canny(img_blur, 80, 150)
    cv2.imshow("Canny edge", img_canny)

    # 发现和标识轮廓
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    marks = np.zeros(img_canny.shape, np.int32)
    imageContours = np.zeros(img_canny.shape, np.uint8)

    # 生成初始标识
    index = 0
    compCount = 0
    while (index >= 0):
        marks = cv2.drawContours(marks, contours, index, compCount + 1, 1, 8, hierarchy)
        imageContours = cv2.drawContours(imageContours, contours, index, 255, 1, 8, hierarchy)
        index = hierarchy[0, index, 0]
        compCount = compCount + 1

    # 显示轮廓图
    markshow = cv2.convertScaleAbs(marks)
    cv2.imshow("imageContours", imageContours)
    # 图像分割
    marks = cv2.watershed(image, marks)
    afterWatershed = cv2.convertScaleAbs(marks)
    cv2.imshow("Watershed", afterWatershed);

    # 对每一个区域进行颜色填充
    perImage = np.zeros(image.shape, np.uint8)
    rows, cols = marks.shape
    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            index = marks[i, j]
            if (index == 255):
                perImage[i, j] = (255, 255, 255)
            else:
                perImage[i, j] = genColor(index)

    cv2.imshow("ColorFill", perImage);

    # 分割并填充颜色的结果跟原始图像融合
    wshed = cv2.addWeighted(image, 0.4, perImage, 0.6, 0)
    cv2.imshow("Added Image", wshed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
