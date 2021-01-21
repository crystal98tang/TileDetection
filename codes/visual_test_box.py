import csv
import cv2
import os
from codes.core.config import cfg
from codes.core.utils import read_csv, draw


if __name__ == '__main__':
    img_path = cfg.PATH.patch_path # "../user_data/Temp_data/train_img"
    anno_path = cfg.PATH.annotation_path # "../user_data/Temp_data/train1.csv"
    save_path = "../user_data/Temp_data/visual_box/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #
    lines = read_csv(anno_path)
    draw(lines, read_path=img_path, save_path=save_path, type='bmp')
    print('successful!')

