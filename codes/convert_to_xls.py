# -*- coding: utf-8 -*-
import os
import time

import cv2
import pandas as pd

from codes.core.config import cfg

class_name_dic = cfg.class_name_dic
lable_color = cfg.lable_color
#
count_error = [0,0,0,0,0,0,0]
output_label_img_mode = False
Statistics_mode = True
time_start = 0
rawImgDir='../tcdata/tile_round1_train_20201231/train_imgs/'
rawLabelFile="../tcdata/tile_round1_train_20201231/train_annos.json"
anno_xls_dir='../user_data/label_xls/'
anno_label_dir='../user_data/label_img/'
#
if Statistics_mode:
    time_start = time.time()  # 开始计时
if not os.path.exists(anno_xls_dir):
    os.makedirs(anno_xls_dir)
if not os.path.exists(anno_label_dir):
    os.makedirs(anno_label_dir)
#
df = pd.read_json(path_or_buf=rawLabelFile, orient='records')   # 读原始json
#
print(df)
#
image_ann={}    # { img_name : [label1, label2, ……]}
for tup in df.itertuples():
    name = tup[5]
    if name not in image_ann:
        image_ann[name] = []
    image_ann[name].append(tup)     # 加
#
for name in image_ann.keys():
    indexs = image_ann[name]
    height, width = indexs[0][3], indexs[0][4]
    #
    if output_label_img_mode:
        im = cv2.imread(os.path.join(rawImgDir, name))
    #
    for tup in indexs:
        category = tup[2]
        bbox = tup[1]
        xmin, ymin, xmax, ymax = bbox
        class_name = class_name_dic[str(category)]
        # 统计
        if Statistics_mode:
            count_error[category] += 1
        # 画框标记
        if output_label_img_mode:
            cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), lable_color[str(category)], 2)
            cv2.putText(im, class_name, (int(xmin), int(ymin) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lable_color[str(category)],2)
    #
    (filename, jpg) = os.path.splitext(name)
    dataframe = pd.DataFrame(indexs)
    writer = pd.ExcelWriter(os.path.join(anno_xls_dir, filename + '.xls'))
    dataframe.to_excel(writer, float_format='%.5f')
    writer.save()
    #
    if output_label_img_mode:
        path = os.path.join(anno_label_dir, name)
        cv2.imwrite(os.path.join(anno_label_dir, name), im)

if Statistics_mode:
    print(count_error)
    time_end = time.time() #计时结束
    print(time_start - time_end)