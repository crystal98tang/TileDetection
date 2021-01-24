'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
import os
from codes.core.yolov3 import YOLO
from core.config import cfg
import  tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

yolo = YOLO()
source_path = cfg.PATH.origin_test_img_path # "../tcdata/tile_round1_train_20201231/train_imgs/"  # 图片来源路径
file_list = os.listdir(source_path)
# img_anno = read_anno_json(rawLabelFile)
frame_size = cfg.input_size
for i,img in tqdm.tqdm(enumerate(file_list)):
    img_name = os.path.split(img)[-1].split('.')[0]
    image = cv2.imread(img,cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR).astype(np.float32)
    raw_image = image.copy()
    raw_h,raw_w = image.shape[:2]
    #判断h方向以及w方向有多少frame
    row = raw_h//frame_size +1  # 5000 / 416 = row
    col = raw_w//frame_size +1  # 5000 / 416 = col
    #确定需要填充的像素
    radius_h = row*frame_size -raw_h
    radius_w = col*frame_size -raw_w
    #边界填充向右向下
    image = cv2.copyMakeBorder(image, 0, radius_h, 0, radius_w, cv2.BORDER_REFLECT)
    image = cv2.copyMakeBorder(image, 0, cfg.gap, 0, cfg.gap, cv2.BORDER_REFLECT)
    sample = raw_image.copy()
    boxes_, scores_, labels_ = [], [], []
    for i in tqdm.tqdm(range(row)):
        for j in range(col):
            image1 = image.copy()
            #相交部分为gap
            subImg = image1[i * frame_size:(i + 1) * frame_size + cfg.gap,
                         j * frame_size:(j + 1) * frame_size + cfg.gap, :]
            subImg /= 255.0
            subImg = np.transpose(subImg, (2, 0, 1))#c h w
            #预测
            predictions,box = yolo.detect_image(image)
            index = 0
            # subImg = subImg.transpose(1, 2, 0)
    #         boxes, scores, labels = run_wbf(predictions, image_index=index, image_size=cfg.image_size)
    #         print(labels)
    #         boxes = boxes.astype(np.int32).clip(min=0, max=cfg.image_size - 1)
    #         boxes[:, 0] = boxes[:, 0] + j * frame_size
    #         boxes[:, 1] = boxes[:, 1] + i * frame_size
    #         boxes[:, 2] = boxes[:, 2] + j * frame_size
    #         boxes[:, 3] = boxes[:, 3] + i * frame_size
    #         boxes_.append(boxes)
    #         scores_.append(scores)
    #         labels_.append(labels)
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # all_annotations = np.array(
    #     [[box[0], box[1], box[2], box[3], score, label] for box, score, label in zip(boxes, scores, labels)])
    #
    # # 丢弃原图像边界外的框
    # keep = (all_annotations[:, 0] < raw_w) & (all_annotations[:, 1] < raw_h)
    # result_annotations = all_annotations[keep]
    # # 限制xmax和ymax的值
    # result_annotations[:, 2] = np.clip(result_annotations[:, 2], 0, raw_w)
    # result_annotations[:, 3] = np.clip(result_annotations[:, 3], 0, raw_h)
    #
    # for ann in result_annotations:
    #     color = distinct_colors[int(ann[5])]
    #     cv2.rectangle(sample, (int(ann[0]), int(ann[1])), (int(ann[2]), int(ann[3])), color, 3)
    #     text_location = (int(ann[0]) + 2, int(ann[1]) - 4)
    #     key = get_key(cfg.class_dict, ann[5])[0]
    #     sample = cv2.putText(sample, f'{key} {ann[4]*100:.2f}%', text_location, font,
    #                              fontScale=0.5, color=color)

    plt.imshow(sample.astype(np.uint8))
    plt.show()
while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()

yolo.close_session()
