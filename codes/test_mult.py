import sys

sys.path.append(r'./codes/core/')
##
from PIL import Image
import os
import tqdm
import json
import cv2
import numpy as np
from core.yolov3 import YOLO_batch
from core.config import cfg
from core.utils import side_black_cut, split_slide, draw

yolo = YOLO_batch()
source_path = cfg.PATH.origin_test_img_path  # "../tcdata/tile_round1_train_20201231/train_imgs/"  # 图片来源路径
file_list = os.listdir(source_path)

final_results = []

for img_name in file_list:
    o_image = cv2.imread(os.path.join(source_path, img_name), 1)  # 读取原始
    img, x_offset, y_offset = side_black_cut(o_image)  # 切黑边
    patch_list, xy_offset_list = split_slide(img, cfg.TEST.patch_size, cfg.TEST.gap)  # 切Patch
    print(img_name + ' ' + str(len(patch_list)) + ' ' + str(x_offset) + ' ' + str(y_offset))
    predict = yolo.detect_image_batch(patch_list, xy_offset_list)  # 预测

    bbox_c_s = []
    for p in predict:
        bbox, categroy, score = p
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = xmin + y_offset, ymin + x_offset, xmax + y_offset, ymax + x_offset  # 黑边坐标转换
        # bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        if cfg.TEST.visual_show:
            bbox_c_s.append([int(xmin), int(ymin), int(xmax), int(ymax), categroy, score])

    if cfg.TEST.visual_show:
        # o_image = cv2.cvtColor(np.asarray(o_image), cv2.COLOR_RGB2BGR)
        cv_img = cv2.imread(os.path.join(source_path, img_name))
        predict_img = draw(cv_img, bbox_c_s, name=False)
        cv2.imwrite(os.path.join(cfg.TEST.visual_save_path, img_name), predict_img)

        # {
        #     "name": "226_46_t20201125133518273_CAM1.jpg",
        #     "category": 4,
        #     "bbox": [
        #         5662,
        #         2489,
        #         5671,
        #         2497
        #     ],
        #     "score": 0.130577
        # },

# with open('result.json', 'w') as fp:
#     json.dump(final_results, fp, indent=4, ensure_ascii=False)
