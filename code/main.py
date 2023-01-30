import os
import pathlib
import cv2 as cv
import csv2img
import numpy as np
import Text_Find
import text_detection


def img_preprocess():
    import pathlib

    # 构建一个pathlib.Path()对象
    # 指定一个目录起始点："F:/path_test"
    p = pathlib.Path("../data")
    # ......
    # 遍历 F:/path_test目录及子目录下 所有的 py 文件
    # 递归遍历所有子目录
    ret = p.glob("**/ProfileData2.csv")
    i = 1
    for item in ret:
        print('\nLoading: ')
        print(item)
        csv2img.csv2img(item, i)
        i = i + 1
    print("Preprocess Done.")


def remove_black_border_img(img):
    H, W = img.shape
    h = 1
    temp_array = np.zeros((1, W))
    while h < H:
        compare_array = img[h, :]
        if not (np.array(compare_array) <= 100).all():
            temp_array = np.concatenate((temp_array, img[h, :].reshape(1, W)), axis=0)
        h += 1
    return temp_array


def flip():
    p = pathlib.Path("../images")
    ret = p.glob("*.png")
    count = 1
    for item in ret:
        img = cv.imread(str(item), 0)
        img = cv.transpose(img)
        # img = cv.flip(img, -1)
        cv.imwrite("../Train_Pic/picture_" + str(count) + ".PNG", img)
        count += 1


if __name__ == '__main__':
    p = pathlib.Path("../Original_Images")
    ret = p.glob("*.PNG")
    num = 1
    for item in ret:
        # Text_Find.find_text_MSER(str(item))
        im1 = text_detection.text_detection_FAST(str(item), num)
        im2 = text_detection.text_detection_SLOW(str(item), num)
        image_v = cv.vconcat([im1, im2])
        output_path = r'../Result/' + 'Result_' + str(num) + '.jpg'
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image_v, "Up: FAST OCR Down: High Acc OCR", (100, 200), font, 5, (0, 0, 255), 5)
        cv.imwrite(output_path, image_v)
        num += 1

