import cv2
import os
import numpy as np
import pandas as pd
import csv
import xlrd
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


def get_bbox(xls_path):
    BBGT = []
    f = pd.read_excel(xls_path)
    num = len(f['bbox'])
    for i in range(num):
        list_img = list(eval(f['bbox'][i]))
        Xmin, Ymin, Xmax, Ymax = int(list_img[0]), int(list_img[1]), int(list_img[2]), int(list_img[3])
        category = int(f["category"][i])
        BBGT.append([Xmin, Ymin, Xmax, Ymax, category])
    return np.array(BBGT)


def split(imgname, dirsrc, dirdst, subsize, gap, iou_thresh=0.3, ext='.bmp'):
    """
    imgname:   待裁切图像名（带扩展名）
    dirsrc:    待裁切的图像保存目录的上一个目录，默认图像与标注文件在一个文件夹下，图像在images下，标注在labelTxt下，标注文件格式为每行一个gt,
               格式为xmin,ymin,xmax,ymax,class,想读其他格式自己动手改
    dirdst:    裁切的图像保存目录的上一个目录，目录下有images,labelTxt两个目录分别保存裁切好的图像或者txt文件，
               保存的图像和txt文件名格式为 oriname_min_ymin.png(.txt),(xmin,ymin)为裁切图像在原图上的左上点坐标,txt格式和原文件格式相同
    subsize:   裁切图像的尺寸，默认为正方形，想裁切矩形自己动手改
    gap:       相邻行或列的图像重叠的宽度
    iou_thresh:小于该阈值的BBGT不会保存在对应图像的txt中（在图像过于边缘或与图像无交集）
    ext:       保存图像的格式
    """
    img = cv2.imread(os.path.join(dirsrc, 'train_imgs', imgname), -1)
    txt_path = os.path.join(os.path.join(dirsrc, 'label_xls'), imgname.split('.')[0] + '.xls')
    csv_path = os.path.join(dirdst, 'train_anno' + '.csv')
    BBGT = get_bbox(txt_path)
    img_h, img_w = img.shape[:2]
    top = -gap
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = -gap
        if top + subsize >= img_h:
            reachbottom = True
            top = max(img_h - subsize, 0)
        while not reachright:
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
                print(len(BBpatch))
                img_name = imgname.split('.')[0]
                cv2.imwrite(os.path.join(os.path.join(dirdst, 'Images'),
                                         img_name + '_' + str(left) + '_' + str(top) + ext), imgsplit)
                lines = []
                for bb in BBpatch:
                    x1, y1, x2, y2, target_id = int(bb[0]) - left, int(bb[1]) - top, int(bb[2]) - left, int(
                        bb[3]) - top, int(bb[4])
                    lines.append([img_name, left, top, x1, y1, x2, y2, target_id])
                with open(csv_path, 'a+', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(lines)
            left += subsize - gap
        top += subsize - gap


if __name__ == '__main__':
    import tqdm

    dirsrc = '../tcdata/tile_round1_train_20201231/'  # 待裁剪图像所在目录的上级目录，图像在images文件夹下，标注文件在labelTxt下
    dirdst = '../user_data/Temp_data/train_img_mult_cutted_total/'  # 裁剪结果存放目录，格式和原图像目录一样
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'Images')):
        os.mkdir(os.path.join(dirdst, 'Images'))
    if not os.path.exists(os.path.join(dirdst, 'Anotations')):
        os.mkdir(os.path.join(dirdst, 'Anotations'))

    subsize = 416
    gap = 104
    iou_thresh = 0.4
    ext = '.bmp'
    num_thresh = 8

    imgnameList = os.listdir(os.path.join(dirsrc, 'train_imgs'))
    for imgname in tqdm.tqdm(imgnameList):
        split(imgname, dirsrc, dirdst, subsize, gap, iou_thresh, ext)
