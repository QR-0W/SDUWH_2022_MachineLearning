# -*- coding: UTF-8 -*-

from PIL import Image
import os


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            if "txt" in filename:
                Filelist.append(os.path.join(home, filename))
                # Filelist.append(filename)  # 文件名列表，只包含文件名

    return Filelist


if __name__ == "__main__":
    filePath = r'../Train_Pic/image'
    outputPath = r'../Train_Pic/txt'
    Filelist = get_filelist(filePath)
    print(len(Filelist))
    # 迭代所有图片
    for filename in Filelist:
        print(filename)
        imgfilename = filename.replace(".txt", ".PNG")
        # 读取图像 标签
        im = Image.open(imgfilename)
        (width, height) = im.size
        # 保存
        output_path = filename.replace(filePath, outputPath)
        outputdir = output_path.rsplit('\\', 1)[0]
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        file_lineinfo = open(output_path, 'w', encoding='utf-8')

        f = open(filename, encoding='utf-8-sig')
        x_1 = y_1 = x_2 = y_2 = x_3 = y_3 = x_4 = y_4 = 0
        x = y = w = h = 0
        left = right = top = bottom = 0
        for line in f.readlines():
            print(line)
            data = line.replace('\n', '')
            substr = data.split(' ')
            x = float(substr[1])
            y = float(substr[2])
            w = float(substr[3])
            h = float(substr[4])
            left = int(width * (x - w / 2))
            right = int(width * (x + w / 2))
            top = int(height * (y - h / 2))
            bottom = int(height * (y + h / 2))
            x_1 = left
            y_1 = top
            x_2 = right
            y_2 = top
            x_3 = right
            y_3 = bottom
            x_4 = left
            y_4 = bottom
            # content = substr[8]
            line_info = [str(x_1), ',', str(y_1), ',', str(x_2), ',', str(y_2), ',', str(x_3), ',', str(y_3), ',',
                         str(x_4), ',', str(y_4), ',', "location", '\n']
            # line_info = [str(label), x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, '\n']
            file_lineinfo.writelines(line_info)
        f.close()
        # line_info = [label, ' ', x_1, ' ', y_1, ' ',  x_2, ' ', y_2, ' ', x_3, ' ', y_3, ' ', x_4, ' ', y_4, '\n']

        file_lineinfo.close()
