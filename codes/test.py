import sys

sys.path.append(r'./codes/core/')
##
from PIL import Image
import os
import tqdm
import json
import cv2
import numpy as np
from core.yolov3 import YOLO
from core.yolov4 import YOLO_v4
from core.config import cfg
from core.utils import side_black_cut, split_slide, draw

yolo = YOLO_v4()
source_path = "../tcdata/tile_round1_testA_20201231/testA_img"  # 图片来源路径
save_json_path = "../user_data/result/"
visual_save_path = "../user_data/result_visual/"

if not os.path.exists(save_json_path):
    os.mkdir(save_json_path)
if not os.path.exists(visual_save_path):
    os.mkdir(os.path.join(visual_save_path))

file_list = os.listdir(source_path)
i = 0
batch = 100

cnt = 0
for img_name in tqdm.tqdm(file_list[batch * i: batch * (i + 1)]):
    cnt += 1
    result = []
    o_image = cv2.imread(os.path.join(source_path, img_name), 1)  # 读取原始
    img, x_offset, y_offset = side_black_cut(o_image)  # 切黑边
    patch_list, xy_offset_list = split_slide(img, cfg.TEST.patch_size, cfg.TEST.gap)  # 切Patch
    print(img_name + ' ' + str(len(patch_list)) + ' ' + str(x_offset) + ' ' + str(y_offset))
    predict = yolo.detect_image(patch_list, xy_offset_list)  # 预测

    bbox_c_s = []
    for p in predict:
        bbox, categroy, score = p
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = xmin + y_offset, ymin + x_offset, xmax + y_offset, ymax + x_offset  # 黑边坐标转换
        # bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        bbox_c_s.append([int(xmin) + 1, int(ymin) + 1, int(xmax) + 1, int(ymax) + 1, categroy + 1, score])  # TODO: +1

    if cfg.TEST.visual_show:
        o_image = cv2.cvtColor(np.asarray(o_image), cv2.COLOR_RGB2BGR)
        cv_img = cv2.imread(os.path.join(source_path, img_name))
        predict_img = draw(cv_img, bbox_c_s, name=False)
        cv2.imwrite(os.path.join(visual_save_path, str(cnt + batch * i) + '.jpg'), img) # TODO:Debug
    #
    if cfg.TEST.out_result:
        for ann in bbox_c_s:
            dict_ann = {}
            # 设置图片name
            # 将图片id对应为name
            dict_ann["name"] = str(img_name)
            # 设置类别category
            dict_ann["category"] = int(ann[4])
            # 设置bbox
            bbox = [ann[0], ann[1], ann[2], ann[3]]
            dict_ann["bbox"] = bbox
            # 设置置信度score
            dict_ann["score"] = np.float(ann[5])
            result.append(dict_ann)

    path = os.path.join(save_json_path, img_name.split(".")[0] + '.json')
    with open(path, 'w') as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)


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
