import cv2
import os
import datetime
import tqdm
import pandas as pd
import numpy as np
import csv
from codes.core.config import cfg


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
        while not reachright:   # 未到右部边界
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
            if len(BBpatch) > 0:  # abandaon images with 0 bboxes
                img_name = imgname.split('.')[0]
                cv2.imwrite(os.path.join(os.path.join(dirdst, 'Images'),
                                         img_name + '_' + str(left) + '_' + str(top) + ext), imgsplit)
                lines = []
                for bb in BBpatch:
                    x1, y1, x2, y2, target_id = int(bb[0]) - left, int(bb[1]) - top, int(bb[2]) - left, int(
                        bb[3]) - top, int(bb[4])
                    lines.append([img_name+'_'+str(left)+'_'+str(top), x1, y1, x2, y2, target_id])
                csv_path = os.path.join(dirdst, 'Anotations',
                                        img_name + '_' + str(left) + '_' + str(top) + '.csv')
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(lines)
            left += subsize - gap
        top += subsize - gap


def change_size(read_file):
    image = cv2.imread(read_file, 1)  # 读取图片
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)  # 阈值 >60 转为 255
    binary_image = b[1]  # 二值图（三通道）
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    img_ary = np.array(binary_image)
    edges_x = []
    edges_y = []
    cnt = 0
    """
    0------bottom------> row
    |
    |left             right
    |
    v
    col      top
    """
    for col in img_ary:     # 遍历列
        cnt += 1
        if col.max() == 255:
            edges_x.append(cnt)
    cnt = 0
    for row in img_ary.T:   # 遍历行
        cnt += 1
        if row.max() == 255:
            edges_y.append(cnt)
    # 老方法 慢掉牙了 -------------
    #
    # x = binary_image.shape[0]
    # y = binary_image.shape[1]
    # edges_x = []
    # edges_y = []
    # for i in range(x):
    #     for j in range(y):
    #         if binary_image[i][j] == 255:
    #             edges_x.append(i)
    #             edges_y.append(j)
    # ------------------------------
    left, right = min(edges_x), max(edges_x)  # 左边界 右边界
    width = right - left  # 宽度
    bottom, top = min(edges_y), max(edges_y)  # 底部 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture, left, bottom  # 返回图片数据、截取的顶部和左边界尺寸


def run(file_list, image_anno, batch, i):
    for name in tqdm.tqdm(file_list[batch * i: batch * (i + 1)]):
        indexs = image_anno[name]
        #
        im, cut_x, cut_y = change_size(source_path + name)
        BBGT = []
        for tup in indexs:
            bbox, category = tup[1], tup[2]
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = xmin - cut_y, ymin - cut_x, xmax - cut_y, ymax - cut_x
            BBGT.append([int(xmin), int(ymin), int(xmax), int(ymax), int(category)])
        split(im, name, np.array(BBGT), cfg.PATH.mult_patch_path, 416, 208)


def read_anno_json():
    df = pd.read_json(path_or_buf=rawLabelFile, orient='records')  # 读原始json
    image_ann = {}  # { img_name : [label1, label2, ……]}
    for tup in df.itertuples():
        name = tup[5]
        if name not in image_ann:
            image_ann[name] = []
        image_ann[name].append(tup)  # 加
    return image_ann


if __name__ == '__main__':
    source_path = cfg.PATH.origin_train_img_path    # "../tcdata/tile_round1_train_20201231/train_imgs/"  # 图片来源路径
    rawLabelFile = "../tcdata/tile_round1_train_20201231/train_annos.json"
    dirdst = cfg.PATH.mult_patch_path
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'Images')):
        os.mkdir(os.path.join(dirdst, 'Images'))
    if not os.path.exists(os.path.join(dirdst, 'Anotations')):
        os.mkdir(os.path.join(dirdst, 'Anotations'))
    file_list = os.listdir(source_path)
    img_anno = read_anno_json()
    i = 0
    batch = 5387
    run(file_list, img_anno, batch, i)
