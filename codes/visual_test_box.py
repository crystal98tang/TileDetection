import csv
import cv2
import os
from codes.core.config import cfg
from codes.core.utils import read_csv, read_and_draw

"""
    批量可视化预测框
"""
if __name__ == '__main__':
    img_path = "../user_data/Temp_data/train_img_mult_cutted_fin/Images"  # 原始图片
    anno_path = "../user_data/Temp_data/train_img_mult_cutted_fin/Anotations"    # 标注索引csv
    save_path = "../user_data/Temp_data/train_img_mult_cutted_fin/visual_box"    # 可视化检测框-存储目录
    #
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #
    list_csv = os.listdir(anno_path)

    read_and_draw(list_csv, anno_path, read_path=img_path, save_path=save_path, type='bmp')    # 画框
    print('successful!')

