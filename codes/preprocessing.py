import sys
sys.path.append(r'./codes/core/')
##
import cv2
import os
import tqdm
import pandas as pd
import numpy as np
import csv
from core.config import cfg
from core.utils import side_black_cut

def BBGT_iou(BBGT, imgRect):
    """
    并不是真正的iou。计算每个BBGT和图像块所在矩形区域的交与BBGT本身的的面积之比，比值范围：0~1
    输入：BBGT：n个标注框，大小为n*4,每个标注框表示为[xmin,ymin,xmax,ymax]，类型为np.array
          imgRect：裁剪的图像块在原图上的位置，表示为[xmin,ymin,xmax,ymax]，类型为np.array
    返回：每个标注框与图像块的iou（并不是真正的iou），返回大小n,类型为np.array
    """
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])
    wh = np.maximum(right_bottom - left_top, 0)
    inter_area = wh[:, 0] * wh[:, 1]
    iou = inter_area / ((BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]))
    return iou


def split(img, imgname, BBGT, dirdst, subsize, gap, iou_thresh=0.8, ext='.bmp'):
    """
    img:       待裁切图像
    imgname:   待裁切图像名（带扩展名）
    BBGT:       n个标注框
    dirdst:    裁切的图像保存目录的上一个目录
    subsize:   裁切图像的尺寸，默认为正方形，想裁切矩形自己动手改
    gap:       相邻行或列的图像重叠的宽度
    iou_thresh:小于该阈值的BBGT不会保存在对应图像的csv中（在图像过于边缘或与图像无交集）
    ext:       保存图像的格式
    """
    img_h, img_w = img.shape[:2]
    top = 0
    reachbottom = False
    while not reachbottom:  # 未到底部边界
        reachright = False
        left = 0
        if top + subsize >= img_h:
            reachbottom = True
            top = max(img_h - subsize, 0)
        while not reachright:  # 未到右部边界
            if left + subsize >= img_w:
                reachright = True
                left = max(img_w - subsize, 0)
            imgsplit = img[top:min(top + subsize, img_h), left:min(left + subsize, img_w)]
            if imgsplit.shape[:2] != (subsize, subsize):
                template = np.zeros((subsize, subsize, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template
            imgrect = np.array([left, top, left + subsize, top + subsize]).astype('float32')
            ious = BBGT_iou(BBGT[:, :4].astype('float32'), imgrect)
            BBpatch = BBGT[ious > iou_thresh]

            # 存储
            img_name = imgname.split('.')[0]
            save_name = img_name + '_' + str(left) + '_' + str(top)
            cv2.imwrite(os.path.join(os.path.join(dirdst, 'Images'), save_name + ext), imgsplit)

            if len(BBpatch) > 0:  # abandaon images with 0 bboxes
                lines = []
                for bb in BBpatch:
                    xmin, ymin, xmax, ymax, target_id = int(bb[0]) - left, int(bb[1]) - top, int(bb[2]) - left, \
                                                        int(bb[3]) - top, int(bb[4])
                    lines.append([img_name + '_' + str(left) + '_' + str(top), xmin, ymin, xmax, ymax, target_id])
                csv_path = os.path.join(dirdst, 'Anotations', save_name + '.csv')

                with open(csv_path, 'w', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(lines)

            left += subsize - gap
        top += subsize - gap


def run(file_list, image_anno, batch, i):
    """
    处理 原始图片--->Patch
    :param file_list: 原始训练集图片
    :param image_anno: 原始标签
    :param batch: 批处理数量
    :param i: 第i轮批处理
    """
    for name in tqdm.tqdm(file_list[batch * i: batch * (i + 1)]):
        indexs = image_anno[name]
        #
        image = cv2.imread(source_path + name, 1)  # 读取图片
        im, x_offset, y_offset = side_black_cut(image)    # 切黑边
        BBGT = []
        for tup in indexs:
            bbox, category = tup[1], tup[2]
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = xmin - y_offset, ymin - x_offset, xmax - y_offset, ymax - x_offset  # 坐标转换
            BBGT.append([int(xmin), int(ymin), int(xmax), int(ymax), int(category)])
        split(im, name, np.array(BBGT), cfg.PATH.mult_patch_path, 416, 208)


def read_anno_json(src):
    df = pd.read_json(path_or_buf=src, orient='records')  # 读原始json
    image_ann = {}  # { img_name : [label1, label2, ……]}
    for tup in df.itertuples():
        name = tup[5]
        if name not in image_ann:
            image_ann[name] = []
        image_ann[name].append(tup)  # 加
    return image_ann


if __name__ == '__main__':
    source_path = cfg.PATH.origin_train_img_path  # "../tcdata/tile_round1_train_20201231/train_imgs/"  # 图片来源路径
    rawLabelFile = "../tcdata/tile_round1_train_20201231/train_annos.json"
    dirdst = "../user_data/Temp_data/train_img_mult_cutted_total"  # cfg.PATH.mult_patch_path
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'Images')):
        os.mkdir(os.path.join(dirdst, 'Images'))
    if not os.path.exists(os.path.join(dirdst, 'Anotations')):
        os.mkdir(os.path.join(dirdst, 'Anotations'))
    file_list = os.listdir(source_path)
    img_anno = read_anno_json(rawLabelFile)
    i = 0
    batch = 3000
    run(file_list, img_anno, batch, i)
