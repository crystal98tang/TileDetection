import cv2
import os
import datetime
import tqdm
import threading


def change_size(read_file):
    image = cv2.imread(read_file, 1)  # 读取图片
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)  # 阈值 >60 转为 255
    binary_image = b[1]  # 二值图（三通道）
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left, right = min(edges_x), max(edges_x)  # 左边界 右边界
    width = right - left  # 宽度
    bottom, top = min(edges_y), max(edges_y)  # 底部 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture, left, top  # 返回图片数据、截取的顶部和左边界尺寸


def run(file_list):
    for i in tqdm.tqdm(range(len(file_list))):
        img, x, y = change_size(source_path + file_names[i])
        cv2.imwrite(save_path + str(x) + '_' + str(y) + '_' + file_names[i], img)


source_path = "G:\TileDetection/tcdata/tile_round1_train_20201231/train_imgs/"  # 图片来源路径
save_path = "G:\TileDetection/tcdata/out/"  # 图片修改后的保存路径

if not os.path.exists(save_path):
    os.mkdir(save_path)

file_names = os.listdir(source_path)
run(file_names[1601:2000])
starttime = datetime.datetime.now()
print("裁剪完毕")
endtime = datetime.datetime.now()  # 记录结束时间
endtime = (endtime - starttime).seconds
print("裁剪总用时", endtime)
