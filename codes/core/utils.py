import numpy as np
import csv
import cv2
import os
from PIL import Image
from tqdm import tqdm

from functools import reduce
from core.config import cfg


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def get_data(anno_line, type='bmp'):
    img_name = anno_line[0] + '.' + type
    img = cv2.imread(os.path.join(cfg.PATH.patch_path, img_name)) / 255
    box = list(eval(anno_line[2]))
    box.append(int(anno_line[1]) - 1)  # FIXME:注意标记类是1~6
    return img, box


def get_random_data_mult(anno_file, max_boxes=100, type='bmp'):
    """
    训练集
    :param annotation_line:
    :param max_boxes:
    :param jitter:
    :param hue:
    :param sat:
    :param val:
    :param random:
    :return:
    """
    anno_lines = read_csv(os.path.join(os.path.join(cfg.PATH.mult_patch_path, "Anotations"), anno_file))
    img_name = anno_lines[0][0] + '.' + type
    image = Image.open(os.path.join(os.path.join(cfg.PATH.mult_patch_path, "Images"), img_name))
    iw, ih = image.size
    h, w = cfg.TRAIN.input_size
    bboxs = []
    for i in anno_lines:
        name, xmin, ymin, xmax, ymax, category = i
        bbox = [int(xmin), int(ymin), int(xmax), int(ymax), int(category) - 1]
        bboxs.append(bbox)
    box = np.array([np.array(bbox) for bbox in bboxs])  # TODO:可能会有误

    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32) / 255
    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    return image_data, box_data


def get_random_data_one(anno_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True,
                        type='bmp'):
    """
    训练集
    :param annotation_line:
    :param input_shape:
    :param max_boxes:
    :param jitter:
    :param hue:
    :param sat:
    :param val:
    :param random:
    :return:
    """
    img_name = anno_line[0] + '.' + type
    image = Image.open(os.path.join(cfg.PATH.patch_path, img_name))
    iw, ih = image.size
    h, w = input_shape
    box = list(eval(anno_line[2]))
    box.append(int(anno_line[1]) - 1)  # FIXME:注意标记类是1~6
    box = np.array([np.array(box)])  # fixme: 目前是设置的是单张patch只有一个框，后续要修改

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32) / 255

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def read_csv(path):
    """
    读取csv
    :param path: 标签csv文件
    :return: 返回list
    """
    with open(path, "r") as f:
        f_csv = csv.reader(f)
        lines = [row for row in f_csv]
    return lines


def draw(list_csv, anno_path, read_path, save_path, type='bmp'):
    """
    标记预测框并保存
    :param anno_lines: 标记索引 list[ 'img_name','left','top','x1','y1','x2','y2','catagory' ]
    :param read_path: 原始图片路径
    :param save_path: 处理后保存路径
    :param type: 图片格式
    :return: None
    """
    for csv in tqdm(list_csv):
        anno_lines = read_csv(os.path.join(anno_path, csv.split(".")[0] + '.csv'))  # 读取标注csv
        im = cv2.imread(os.path.join(read_path, csv.split(".")[0] + '.' + type))
        for i in anno_lines:
            name, xmin, ymin, xmax, ymax, category = i
            #
            class_name = cfg.class_name_dic[str(category)]
            # 画框标记
            cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), cfg.lable_color[str(category)], 2)
            cv2.putText(im, class_name, (int(xmin), int(ymin) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        cfg.lable_color[str(category)], 2)
        # 保存
        cv2.imwrite(os.path.join(save_path, name + '.' + type), im)


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
