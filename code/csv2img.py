import pandas as pd
import cv2
import cv2 as cv
import numpy as np
import time


def csv2img(csv_path, number):
    # 载入图像
    img = load_csv(csv_path)

    # 裁切图像
    img = cut_img(img, 7500)

    # 利用加权均值模糊填洞，直到洞被填满
    kernel = np.ones((3, 3), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    # img_mask = cv.blur(img, ksize=(3, 3), borderType=cv.BORDER_REPLICATE)
    # img += img_mask

    # 展平图像
    img1 = flattening(img, radius=200)
    img2 = flattening(img, radius=20)
    img = img1 + img2
    # 将图像缩放至（0，1）之间
    img = normalization(img, threshold=1e5)

    # 转换为SDR图像
    img = convert_to_sdr(img)

    # # 对图像滤波
    # img = filtering_img(img)

    # 直方图裁切
    img = hist_clip(img, 0.5)

    # # 直方图均值化
    # img = cv.equalizeHist(img[:, :, 0])
    # # 均值化光照
    # img = evenlighting_img(img, radius=300)

    # img = pystripe.filter_streaks(img, sigma=[128, 256], level=7, wavelet='db2')

    # 重塑图像
    img = cv.transpose(img)
    img = cv.flip(img, -1)

    # 去除黑边
    img = remove_black_border_img(img)

    # 输出图像
    output_path = r'../Train_Pic/' + 'csv_to_img_' + str(number) + '.PNG'
    cv.imwrite(output_path, img)


def load_csv(csv_path):
    a = time.perf_counter()
    points_data = pd.read_csv(csv_path, sep=',', header=None, skiprows=1)
    points_data = points_data.to_numpy()
    points_data = points_data[:, 2:]
    # 存储格式是（X，Z，Y）
    # 读取数据并对其预处理
    points_z = points_data[:, [num for num in range(points_data.shape[1]) if num % 3 == 1]]
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Load Csv Used Time: " + delta_time + "s")
    return points_z


def full_hole_erode(img, kernel_matrix=np.ones((5, 5), np.uint8), it=2):
    # 设置kernel，也可以利用getStructuringElement()函数指明它的形状
    kernel = kernel_matrix
    # 腐蚀
    img = cv2.erode(img, kernel, iterations=it)
    return img


def full_hole_dilate(img, kernel_matrix=np.ones((5, 5), np.uint8), it=2):
    # 设置kernel，也可以利用getStructuringElement()函数指明它的形状
    kernel = kernel_matrix
    # 腐蚀
    img = cv2.dilate(img, kernel, iterations=it)
    return img


def full_img(img, radius):
    i = 1
    while np.count_nonzero(img) != np.size(img):
        delta1 = time.perf_counter()
        img = full_hole(img, radius)
        i += 1
        delta2 = time.perf_counter()
        delta_time = delta2 - delta1
        print("Full Hole, Round: " + str(i) + "Used Time: " + str(delta_time) + "s")
    return img


def full_hole(img, radius):
    imgSum = np.zeros(img.shape)
    imgCount = np.zeros(img.shape, dtype="int32")
    wd = img.shape[1]
    ht = img.shape[0]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            posX = np.clip(np.arange(j, wd + j), 0, wd - 1)
            posY = np.clip(np.arange(i, ht + i), 0, ht - 1)
            posX, posY = np.meshgrid(posX, posY)
            imgSum += img[posY, posX]
            imgCount += np.where(img[posY, posX] == 0, 0, 1)
    imgCount = np.where(imgCount == 0, 1, imgCount)
    img = np.where(img == 0, imgSum / imgCount, img)
    img = img.astype("float32")

    return img


def flattening_img(img):
    a = time.perf_counter()
    img_f = img.astype(int)
    avg_array = np.average(img_f, axis=1)
    for i in range(img_f.shape[0]):
        img_f[i, :] = img_f[i, :] - avg_array[i]

    min_element = min(min(row) for row in img_f)
    img_f = img_f + abs(min_element)
    img_f = np.interp(img_f, (img_f.min(), img_f.max()), (0, 255))

    b = time.perf_counter()
    delta_time = str(b - a)
    print("Flattening Image Used Time: " + delta_time + "s")
    return img_f


def flattening(img, radius):
    a = time.perf_counter()
    img = img - np.average(img, 0)
    blur_mask = cv.blur(img, (radius, radius), borderType=cv.BORDER_REPLICATE)
    img = img - blur_mask
    img = img.astype("float32")
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Flattening Image Used Time: " + delta_time + "s")
    return img


def filtering_img(img):
    fa = time.perf_counter()
    img = np.repeat(img[:, :, np.newaxis], 3, 2)
    img = cv.pyrMeanShiftFiltering(img, 10, 10)
    fb = time.perf_counter()
    delta_time = str(fb - fa)
    print("Filtering Used Time: " + delta_time + "s")
    return img


def normalization(img, threshold):
    # 移除无效点后缩放至0~1之间
    a = time.perf_counter()
    min_h = np.nanmin(img)
    max_h = np.nanmax(img)
    img = np.nan_to_num(img)
    img = (img - min_h) / (max_h - min_h)
    if threshold != 0:
        # 统计像素
        hist = np.histogram(img, bins=np.arange(0, 1.01, 0.01, dtype=np.float))
        min_h = 0.0
        max_h = 1.0
        max_len = 0
        min_h0 = 0
        max_h0 = 100
        in_h = False

        for i in range(len(hist[0])):
            if hist[0][i] >= threshold:
                if not in_h:
                    min_h0 = i
                in_h = True
            elif in_h:
                max_h0 = i
                in_h = False
                if max_h0 - min_h0 > max_len:
                    max_len = max_h0 - min_h0
                    min_h = min_h0 / 100
                    max_h = max_h0 / 100
        img = np.clip((img - min_h) / (max_h - min_h), 0.0, 1.0)

    b = time.perf_counter()
    delta_time = str(b - a)
    print("Normalization Used Time: " + delta_time + "s")
    return img


def convert_to_sdr(img):
    a = time.perf_counter()
    img = (img * 255).astype("uint8")
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Converting SDR Image Used Time: " + delta_time + "s")
    return img


def hist_clip(img, threshold):
    a = time.perf_counter()
    hist = np.histogram(img, bins=np.arange(0, 256, 1, dtype="uint8"))
    clip_point = 0
    sum_pixel = 0
    for i in range(len(hist[0])):
        sum_pixel += hist[0][i]
        if sum_pixel >= threshold * img.shape[0] * img.shape[1]:
            clip_point = i
            break
    img = (np.clip(((img.astype("float32") - clip_point) / (255 - clip_point)), 0, 1) * 255).astype("uint8")
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Hist Clipping Used Time: " + delta_time + "s")
    return img


def cut_img(img, threshold):
    a = time.perf_counter()
    cutL = 0
    cutR = 2047
    for i in range(0, 2048, 1):
        if np.count_nonzero(img[:, i]) > threshold:
            cutL = i
            break
    for i in range(2047, -1, -1):
        if np.count_nonzero(img[:, i]) > threshold:
            cutR = i
            break
    img = img[:, cutL:cutR]
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Cut Image Used Time: " + delta_time + "s")
    return img


def unevenLightCompensate(img, blockSize):
    gray = img
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if rowmax > gray.shape[0]:
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if colmax > gray.shape[1]:
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv.GaussianBlur(dst, (3, 3), 0)

    return dst


def evenlighting_img(img, radius=300):
    a = time.perf_counter()
    img_f = unevenLightCompensate(img, radius) + np.average(img, 0)
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Evenlighting Used Time: " + delta_time + "s")
    return img_f


def remove_black_border_img(img):
    a = time.perf_counter()
    H, W = img.shape
    h = 1
    temp_array = np.zeros((1, W))
    while h < H:
        compare_array = img[h, :]
        if not (np.array(compare_array) <= 20).all():
            temp_array = np.concatenate((temp_array, img[h, :].reshape(1, W)), axis=0)
        h += 1
    b = time.perf_counter()
    delta_time = str(b - a)
    print("Remove Black Border Used Time: " + delta_time + "s")
    return temp_array
    # Y, X = img.shape
    # y = 1
    # tempArray = np.zeros((1, X))
    # while y < Y - 10:
    #     temp = img[y:y + 60, 1]
    #     if not (np.array(temp) < 10).all():
    #         tempArray = np.concatenate((tempArray, img[y, :].reshape(1, X)), axis=0)
    #     y = y + 1
    # return tempArray


def binarization_img(img, blocksize=11, C=-30):
    img_Binary = img
    ret, otsu = cv2.threshold(img_Binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_B = cv.adaptiveThreshold(img_Binary, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blocksize, C)
    # conv_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 生成5x5的全1矩阵
    # img_B = cv.dilate(img_B, conv_kernel)
    return img_Binary
