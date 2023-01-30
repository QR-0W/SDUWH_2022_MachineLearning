# text_detection.py 对图像进行文本检测
import time
import cv2
import numpy
import numpy as np
# 导入必要的包
from imutils.object_detection import non_max_suppression  # 从IMUTIL中导入了NumPy、OpenCV的非最大单位抑制实现
import paddleocr
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import stripe

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def text_detection(path, width=4096, height=960, min_confidence=0.5, east="../model/frozen_east_text_detection.pb"):
    # 加载图像获取维度
    image = cv2.imread(str(path))
    orig = image.copy()
    (H, W) = image.shape[:2]

    # 计算宽度，高度及分别的比率值
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)

    # 缩放图像获取新维度
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # 为了使用OpenCV和EAST深度学习模型执行文本检测，需要提取两层的输出特征图：
    # 定义EAST探测器模型的两个输出层名称，感兴趣的是——第一层输出可能性，第二层用于提取文本边界框坐标
    # 第一层是输出sigmoid激活，提供了一个区域是否包含文本的概率。
    # 第二层是表示图像“几何体”的输出特征映射-将能够使用该几何体来推导输入图像中文本的边界框坐标
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # cv2.dnn.readNet加载预训练的EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # 从图像构建一个blob，然后执行预测以获取俩层输出结果
    # 将图像转换为blob来准备图像
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    # 通过将层名称作为参数提供给网络以指示OpenCV返回感兴趣的两个特征图：
    # 分数图，包含给定区域包含文本的概率
    # 几何图：输入图像中文本的边界框坐标的输出几何图
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # 展示文本预测耗时信息
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # 从分数卷中获取行数和列数，然后初始化边界框矩形集和对应的信心分数
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # 两个嵌套for循环用于在分数和几何体体积上循环，这将是一个很好的例子，说明可以利用Cython显著加快管道操作。
    # 我已经用OpenCV和Python演示了Cython在快速、优化的“for”像素循环中的强大功能。
    # 遍历预测结果
    for y in range(0, numRows):
        # 提取分数（概率），然后是环绕文字的几何（用于推导潜在边界框坐标的数据）
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # 遍历列
        for x in range(0, numCols):
            # 过滤弱检测
            if scoresData[x] < min_confidence:
                continue

            # 计算偏移因子，因为得到的特征图将比输入图像小4倍
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # 提取用于预测的旋转角度，然后计算正弦和余弦
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # 使用几何体体积导出边界框
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # 计算文本边界框的开始，结束x，y坐标
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # 将边界框坐标和概率分数添加到各自的列表
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # 对边界框应用非最大值抑制（non-maxima suppression），以抑制弱重叠边界框，然后显示结果文本预测
    # apply overlapping to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # 遍历边界框
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        # 根据相对比率缩放边界框坐标
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # 在图像上绘制边界框
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig, str(confidences[i]), (startX, startY - 10), font, 1, (0, 255, 0), 2)
        if (0 < endX < orig.shape[0]) & (0 < endY < orig.shape[1]) & (0 < startX < orig.shape[0]) & (
                0 < startY < orig.shape[1]):
            text_to_ocr = orig[startX:endX, startY:endY, :]
            ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
            txts = ""
            result = ocr.ocr(text_to_ocr, det=False, cls=True)
            for line in result:
                txts += str(line)
            pos = ((startX + endX) / 2, (startY + endY) / 2)
            cv2.putText(orig, str(txts), (pos), font, 1, (0, 0, 255), 2)
            print(txts)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 展示输出图像
    cv2.namedWindow("Text Detection", 0)
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def text_detection_FAST(path, num, ):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_mp=True, total_process_num=20,
                              det_model_dir=r"D:\TireOCR\model\export_model", det_db_thresh=0.4, det_algorithm='DB',
                              det_limit_side_len=5000, det_db_score_mode='fast', det_db_unclip_ratio=2.0,
                              det_box_type='quad',
                              rec_model_dir="../model/en_PP-OCRv3_rec_infer", rec_algorithm='CRNN',
                              rec_batch_num=10, max_text_length=30
                              )
    # image_ocr = cv2.imread(str(path), 0)

    image = cv2.imread(str(path))

    # cv2.namedWindow("Binary", 0)
    # cv2.imshow("Binary", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result = ocr.ocr(image, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    from PIL import Image
    result = result[0]

    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    for line in result:
        # [startX, startY], [endX, startY], [endX, endY], [startX, endY]
        box = line[0]
        startX = int(box[0][0])
        startY = int(box[0][1])
        endX = int(box[2][0])
        endY = int(box[2][1])
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)

        txts = line[1][0]
        scores = line[1][1]

        cv2.putText(image, str(txts), (startX, startY + 10), font, 1.2, (0, 0, 255), 2)

    # output_path = r'../Cache/' + 'Result_SVTR_' + str(num) + '.jpg'
    # cv2.imwrite(output_path, image)
    return image


def text_detection_SLOW(path, num):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_mp=True, total_process_num=20,
                              det_model_dir=r"D:\TireOCR\model\export_model", det_db_thresh=0.4, det_algorithm='DB',
                              det_limit_side_len=10000, det_db_score_mode='slow', det_db_unclip_ratio=2.0,
                              det_box_type='quad',
                              rec_model_dir="../model/en_PP-OCRv3_rec_infer", rec_algorithm='SVTR_LCNet',
                              rec_batch_num=10, max_text_length=30
                              )
    # image_ocr = cv2.imread(str(path), 0)

    image = cv2.imread(str(path))
    # image_ocr = cv2.adaptiveThreshold(image_ocr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

    # cv2.namedWindow("Binary", 0)
    # cv2.imshow("Binary", image_ocr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result = ocr.ocr(image, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    from PIL import Image
    result = result[0]

    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    for line in result:
        # [startX, startY], [endX, startY], [endX, endY], [startX, endY]
        box = line[0]
        startX = int(box[0][0])
        startY = int(box[0][1])
        endX = int(box[2][0])
        endY = int(box[2][1])
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 5)

        txts = line[1][0]
        scores = line[1][1]

        cv2.putText(image, str(txts), (startX, startY + 10), font, 1, (0, 255, 0), 2)

    # output_path = r'../Cache/' + 'Result_CRNN_' + str(num) + '.jpg'
    # cv2.imwrite(output_path, image)
    return image
