import csv
import cv2
import os
from codes.core.config import cfg
from codes.core.utils import read_csv, draw

"""
    批量可视化预测框
"""
if __name__ == '__main__':
    img_path = cfg.PATH.patch_path  # 原始图片
    anno_path = cfg.PATH.annotation_path    # 标注索引csv
    save_path = "../user_data/Temp_data/visual_box2/"    # 可视化检测框-存储目录
    #
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #
    lines = read_csv(anno_path)     # 读取标注csv
    draw(lines, read_path=img_path, save_path=save_path, type='bmp')    # 画框
    print('successful!')

