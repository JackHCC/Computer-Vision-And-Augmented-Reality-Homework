import numpy as np
import cv2
import random

Image = cv2.imread('fruit.jpg')
# Image = cv2.imread('../../flower.jpg')
x, y, color_value = Image.shape

# 申请图像空间
feature_matrix = np.zeros((x * y, 5))
segmented_img = np.zeros(Image.shape, )
m = 0
h = 155
iterations = 30
for i in range(0, x):
    for j in range(0, y):
        feature_matrix[m, 0] = Image[i, j, 0]  # color storage
        feature_matrix[m, 1] = Image[i, j, 1]
        feature_matrix[m, 2] = Image[i, j, 2]
        feature_matrix[m, 3] = i  # position storing
        feature_matrix[m, 4] = j
        m = m + 1

# 生成初始中心点
flag = 0
while (len(feature_matrix) > 0):
    index = random.randint(0, len(feature_matrix) - 1)
    # selected a point p if flag 0
    if (flag == 0):
        mean_r = feature_matrix[index, 0]
        mean_g = feature_matrix[index, 1]
        mean_b = feature_matrix[index, 2]
        mean_x = feature_matrix[index, 3]
        mean_y = feature_matrix[index, 4]

    cluster = []
    eucli_dist = 0

    for i in range(0, len(feature_matrix)):
        eucli_dist = np.sqrt((mean_r - feature_matrix[i, 0]) ** 2 + (mean_g - feature_matrix[i, 1]) ** 2 + (
                    mean_b - feature_matrix[i, 2]) ** 2 + (mean_x - feature_matrix[i, 3]) ** 2 + (
                                         mean_y - feature_matrix[i, 4]) ** 2)
        if eucli_dist < h:
            cluster.append(i)
    # 更新
    print("to check in", len(feature_matrix))
    if (len(cluster) > 0):
        new_mean_r = 0
        new_mean_g = 0
        new_mean_b = 0
        new_mean_x = 0
        new_mean_y = 0
        for i in range(0, len(cluster)):
            new_mean_r += feature_matrix[cluster[i]][0]
            new_mean_g += feature_matrix[cluster[i]][1]
            new_mean_b += feature_matrix[cluster[i]][2]
            new_mean_x += feature_matrix[cluster[i]][3]
            new_mean_y += feature_matrix[cluster[i]][4]

        new_mean_r = new_mean_r / len(cluster)
        new_mean_g = new_mean_g / len(cluster)
        new_mean_b = new_mean_b / len(cluster)
        new_mean_x = new_mean_x / len(cluster)
        new_mean_y = new_mean_y / len(cluster)

        new_eucli_dist = np.sqrt(
            (new_mean_r - mean_r) ** 2 + (new_mean_g - mean_g) ** 2 + (new_mean_b - mean_b) ** 2 + (
                        new_mean_x - mean_x) ** 2 + (new_mean_y - mean_y) ** 2)
        print("eucledian dist= ", new_eucli_dist)

        if (new_eucli_dist < iterations):
            print("cluster.length ", len(cluster))
            for i in range(0, len(cluster)):
                x = int(feature_matrix[cluster[i]][3])
                y = int(feature_matrix[cluster[i]][4])
                segmented_img[x][y][0] = new_mean_r
                segmented_img[x][y][1] = new_mean_g
                segmented_img[x][y][2] = new_mean_b
            feature_matrix = np.delete(feature_matrix, cluster, 0)
            flag = 0
            # print feature_matrix.shape

        else:
            mean_r = new_mean_r
            mean_g = new_mean_g
            mean_b = new_mean_b
            mean_x = new_mean_x
            mean_y = new_mean_y

            flag = 1

        print("to check out", feature_matrix.shape)

print("done")
cv2.imshow("final_seg", segmented_img / segmented_img.max())

cv2.waitKey(0)
cv2.destroyAllWindows()
