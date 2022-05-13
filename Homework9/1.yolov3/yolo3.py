import numpy as np
import cv2 as cv
import os
import time

weightsPath = 'yolov3.weights'  # 权重文件
configPath = 'yolov3.cfg'  # 配置文件
labelsPath = 'coco.names'  # label名称
# imgPath = 'zoom.jpg'
imgPath = 'test.jpg'
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值

net = cv.dnn.readNetFromDarknet(configPath, weightsPath)  # 加载网络
print("[INFO] loading YOLO from disk...")

img = cv.imread(imgPath)
blobImg = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True, False)
net.setInput(blobImg)  # 将图片送入输入层

# 获取网络输出层信息（所有输出层的名字），设定并前向传播
outInfo = net.getUnconnectedOutLayersNames()  # yolo在每个scale都有输出，outInfo是每个scale的名字信息
start = time.time()
layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息
end = time.time()
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

(H, W) = img.shape[:2]  # 图像尺寸
# 过滤layerOutputs
# layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
# 过滤后的结果放入：
boxes = []  # 所有边界框（各层结果放一起）
confidences = []  # 所有置信度
classIDs = []  # 所有分类ID

# # 1）过滤掉置信度低的框
for out in layerOutputs:  # 各个输出层
    for detection in out:  # 各个框
        # 获取置信度
        scores = detection[5:]  # 各个类别的置信度
        classID = np.argmax(scores)  # 最高置信度的id即为分类id
        confidence = scores[classID]  # 拿到置信度

        # 根据置信度筛查
        if confidence > CONFIDENCE:
            box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉，保留的box的索引index存入idxs
idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
# 得到labels列表
with open(labelsPath, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
# 应用检测结果
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")  # 框显示颜色
if len(idxs) > 0:
    for i in idxs.flatten():  # indxs是二维的，第0维是输出层，这里把它展平成1维
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color,
                   2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px
cv.imshow('detected image', img)
cv.waitKey(0)
cv.destroyAllWindows()
