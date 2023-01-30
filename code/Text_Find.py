# -*- coding: utf-8 -*-

import cv2
import numpy as np


def find_text_morphology(morphology_path):
    # 读取图片
    imagePath = morphology_path
    img = cv2.imread(imagePath)

    # 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 利用Sobel边缘检测生成二值图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 膨胀、腐蚀
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    #  查找轮廓和筛选文字区域
    region = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]

        # 计算轮廓面积，并筛选掉面积小的
        area = cv2.contourArea(cnt)
        if (area < 1000):
            continue

        # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)
        print("rect is: ")
        print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 根据文字特征，筛选那些太细的矩形，留下扁的
        if (height > width * 1.3):
            continue

        region.append(box)

    # 绘制轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_text_MSER(mesr_path):
    # 读取图片
    imagePath = mesr_path
    img = cv2.imread(imagePath)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    orig = img.copy()

    # 调用 MSER 算法
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    cv2.imshow('img', img)

    # 将不规则检测框处理成矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
        if (10000 < w * h < 250000):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
    cv2.namedWindow("hulls", 0)
    cv2.imshow("hulls", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()


# NMS 方法（Non Maximum Suppression，非极大值抑制）
def nms(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # 取四个坐标数组
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算面积数组
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按得分排序（如没有置信度得分，可按坐标从小到大排序，如右下角坐标）
    idxs = np.argsort(y2)

    # 开始遍历，并删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 找剩下的其余框中最大坐标和最小坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算重叠面积占对应框的比例，即 IoU
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # 如果 IoU 大于指定阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")


if __name__ == '__main__':
    path = "../Train_Pic/image/picture_1.PNG"
    find_text_MSER()
