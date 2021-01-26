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


def contrast_img(img1, c, b):
    """
    调节亮度
    :param img1:
    :param c:
    :param b:
    :return:
    """
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return dst


def split_slide(img, patch_size, gap, up=True):
    """
    滑动裁切
    img:   待裁切的高分辨率图像
    """
    # init_left = -gap  # TODO：追加边缘切图
    # init_top = -gap

    img_h, img_w = img.shape[:2]
    top = 0

    patch_list, xy_offset_list = [], []
    reachbottom = False

    while not reachbottom:
        reachright = False
        left = 0
        if top + patch_size >= img_h:
            reachbottom = True
            top = max(img_h - patch_size, 0)
        while not reachright:
            if left + patch_size >= img_w:
                reachright = True
                left = max(img_w - patch_size, 0)

            imgsplit = img[top:min(top + patch_size, img_h), left:min(left + patch_size, img_w)]

            if imgsplit.shape[:2] != (patch_size, patch_size):
                template = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                template[:, :] = 50
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template

            # cv2.imshow("patch split", imgsplit)   # debug:

            # if up:
            #     imgsplit = contrast_img(imgsplit, 1.2, 3)   # 提升亮度

            patch_list.append(imgsplit)
            xy_offset_list.append([left, top])

            left += patch_size - gap

        top += patch_size - gap

    return patch_list, xy_offset_list


def longestConsecutive(list):
    """
    求最长连续子串
    :param list:
    :return:
    """
    start = 0
    length = 0
    max_start = 0
    max_length = 0
    for i in range(len(list) - 1):
        if list[i] == list[i + 1] - 1:
            length += 1
        else:
            if length > max_length:
                max_start = start
                max_length = length
            start = i
            length = 0
    # 最大子list即原始list
    if length > max_length:
        max_start = start
        max_length = length
    #
    return list[max_start], list[max_start + max_length], max_length


def side_black_cut(image):
    """
    切黑边
    :param image: 图片
    :return: 切后image、所切左边界尺寸、所切上边界尺寸
    """
    # image = np.array(image)
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
    for col in img_ary:  # 遍历列
        cnt += 1
        if col.max() == 255:
            edges_x.append(cnt)
    cnt = 0
    for row in img_ary.T:  # 遍历行
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
    # left, right = min(edges_x), max(edges_x)  # 左边界 右边界
    # width = right - left  # 宽度
    # bottom, top = min(edges_y), max(edges_y)  # 底部 顶部
    # height = top - bottom  # 高度
    # ------------------------------

    left, right, width = longestConsecutive(edges_x)  # 左边界 右边界
    bottom, top, height = longestConsecutive(edges_y)  # 底部 顶部

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    # cv2.imshow("cut black", pre1_picture)     # debug:
    return pre1_picture, left, bottom  # 返回图片数据、截取的顶部和左边界尺寸


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


def read_and_draw(list_csv, anno_path, read_path, save_path, type='.bmp'):
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
        name, show_img = draw(im, anno_lines)
        # 保存
        cv2.imwrite(os.path.join(save_path, str(name) + type), show_img)


def draw(im, anno_list, name=True, score=False):
    """
    标记预测框
    :param im: 图片
    :param anno_list: 标记索引 list[ ]
    :return: None
    """
    for i in anno_list:
        if name:
            name, xmin, ymin, xmax, ymax, category = i
            class_name = cfg.class_name_dic[str(category)]
            label = class_name
        else:
            xmin, ymin, xmax, ymax, category, score = i
            class_name = cfg.class_name_dic[str(category)]
            label = '{} {:.2f}'.format(class_name, score)
        # 画框标记
        cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), cfg.lable_color[str(category)], 2)
        cv2.putText(im, label, (int(xmin), int(ymin) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    cfg.lable_color[str(category)], 2)
    if name:
        return name, im
    return im


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


def nms(dets, thresh):
    """
    nms
    :param dets:
    :param thresh:
    :return: 保留的索引
    """
    if len(dets) == 0:
        return []
    dets = np.array(dets)
    # boxes 位置
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # boxes scores
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 各 box 的面积
    order = scores.argsort()[::-1]  # boxes 的按照 score 排序

    keep = []  # 记录保留下的 boxes
    while order.size > 0:
        i = order[0]  # score 最大的 box 对应的 index
        keep.append(i)  # 将本轮 score 最大的 box 的 index 保留

        # 计算剩余 boxes 与当前 box 的重叠程度 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # IoU
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于设定阈值的 boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
