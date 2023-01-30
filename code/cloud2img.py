import cv2
import re
import numpy as np
import torch
from matplotlib import pyplot

import os, sys

os.chdir(sys.path[0])


class verCloud2Img:
    def __init__(self, path: str) -> None:
        self.path = path
        self.img = self.__loadImg(path)  # 点云转二维数组
        self.__Cut(7500)  # 裁剪，参数为有效像素的个数
        while np.count_nonzero(self.img) != np.size(self.img):
            self.__fullHole(2)  # 使用加权均值模糊反复填洞，直到洞被填满
        self.__flatImg(100)  # 使用扫描线均值消除曲率
        self.__normalizeImg(1e5)
        # 转换为SDR图像
        self.img = (self.img * 255).astype("uint8")
        # 滤波
        self.img = np.repeat(self.img[:, :, np.newaxis], 3, 2)
        self.img = cv2.pyrMeanShiftFiltering(self.img, 10, 10)
        # 直方图均衡化
        self.img = cv2.equalizeHist(self.img[:, :, 0])
        # 直方图剪裁
        self.__clipHist(0.5)

    def __normalizeImg(self, thre: int):
        # 移除无效点后规格化到0~1
        # self.img = np.where(self.img == 0, np.nan, self.img)
        minH = np.nanmin(self.img)
        maxH = np.nanmax(self.img)
        self.img = np.nan_to_num(self.img)
        self.img = (self.img - minH) / (maxH - minH)
        if (thre != 0):
            # 像素统计
            hist = np.histogram(self.img, bins=np.arange(0, 1.01, 0.01, dtype=np.float))

            # 基于直方图和给定阈值重新规格化图像(移除两侧的无效点)
            minH = 0.0
            maxH = 1.0
            maxLen = 0

            minH0 = 0
            maxH0 = 100
            inH = False
            for i in range(len(hist[0])):
                if (hist[0][i] >= thre):
                    if (inH == False):
                        minH0 = i
                    inH = True
                elif (inH == True):
                    maxH0 = i
                    inH = False
                    if (maxH0 - minH0 > maxLen):
                        maxLen = maxH0 - minH0
                        minH = minH0 / 100
                        maxH = maxH0 / 100
            self.img = np.clip((self.img - minH) / (maxH - minH), 0.0, 1.0)

    def __Cut(self, thre):
        cutL = 0
        cutR = 2047
        for i in range(0, 2048, 1):
            if (np.count_nonzero(self.img[:, i]) > thre):
                cutL = i
                break
        for i in range(2047, -1, -1):
            if (np.count_nonzero(self.img[:, i]) > thre):
                cutR = i
                break
        self.img = self.img[:, cutL:cutR]

    def __fullHole(self, radius):
        imgSum = np.zeros(self.img.shape)
        imgCount = np.zeros(self.img.shape, dtype="int32")
        wd = self.img.shape[1]
        ht = self.img.shape[0]
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                posX = np.clip(np.arange(j, wd + j), 0, wd - 1)
                posY = np.clip(np.arange(i, ht + i), 0, ht - 1)
                posX, posY = np.meshgrid(posX, posY)
                imgSum += self.img[posY, posX]
                imgCount += np.where(self.img[posY, posX] == 0, 0, 1)
        imgCount = np.where(imgCount == 0, 1, imgCount)
        self.img = np.where(self.img == 0, imgSum / imgCount, self.img)
        self.img = self.img.astype("float32")

    def __loadImg(self, path: str):
        with open(path, "r") as csvFile:
            s = csvFile.readline()  # 读掉表头
            # 读取csv到np矩阵
            img = np.ndarray((8000, 2048), dtype=np.float32)
            for i in range(0, 8000):
                s = csvFile.readline()
                arr = s.split(",")
                arr = arr[2:]
                img[i] = arr[1::3]
        return img

    def __flatImg(self, radius: int):
        self.img = self.img - np.average(self.img, 0)  # 使用扫描线均值消除曲率
        blurImg = cv2.blur(self.img, (radius, radius), borderType=cv2.BORDER_REPLICATE)
        self.img = self.img - blurImg  # 使用均值模糊估计高度消除曲率

    def __clipHist(self, thre: float):
        hist = np.histogram(self.img, bins=np.arange(0, 256, 1, dtype="uint8"))
        clipPoint = 0
        sumPix = 0
        for i in range(len(hist[0])):
            sumPix += hist[0][i]
            if (sumPix >= thre * self.img.shape[0] * self.img.shape[1]):
                clipPoint = i
                break

        self.img = (np.clip(((self.img.astype("float32") - clipPoint) / (255 - clipPoint)), 0, 1) * 255).astype("uint8")


# End of class "vetCloud2Img"

def getFileList(root: str, REpattern: str):
    files = os.listdir(root)
    Pattern = re.compile(REpattern)
    Filterfiles = []
    for file in files:
        if (re.search(Pattern, file) != None):
            Filterfiles.append(file)
    return files


if __name__ == "__main__":
    import pathlib

    p = pathlib.Path("../data")
    for csvFileLists in p.glob('**/ProfileData2.csv'):
        vcImg = verCloud2Img(csvFileLists)

    # csvFileLists = getFileList("data", r".+\.csv$")
    # for csvFile in csvFileLists:
    #     vcImg = verCloud2Img(f"data/{csvFile}")
    #     # 导出贴图
    #     cv2.imwrite(f"OutputImgs/full/{csvFile[:-4]}.png", vcImg.img)
    #     print(f"output:OutputImgs/full/{csvFile[:-4]}.png")
