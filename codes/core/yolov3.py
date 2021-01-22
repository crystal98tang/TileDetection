from functools import wraps

import numpy as np
import tensorflow as tf
import os
import colorsys

from timeit import default_timer as timer
from keras import backend as K
from keras.layers import (Add, Concatenate, Conv2D, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D, Input)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
from codes.core.utils import compose, letterbox_image

from codes.core.darknet53 import darknet_body

from codes.core.config import cfg
from PIL import Image, ImageDraw, ImageFont


class YOLO(object):
    _defaults = {
        "model_path": cfg.PATH.test_model_path,
        "anchors_path": cfg.PATH.anchors_info,
        "classes_path": cfg.PATH.classes_info,
        "score": cfg.TEST.score_threshold,
        "iou": cfg.TEST.iou_threshold,
        "max_boxes": cfg.TEST.max_box_num,
        "model_image_size": cfg.TEST.INPUT_SIZE
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化yolo
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # ---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        # ---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # ---------------------------------------------------------#
        #   载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        #   否则先构建模型再载入
        # ---------------------------------------------------------#
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        # ---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        # ---------------------------------------------------------#
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        start = timer()
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        new_image_size = (self.model_image_size[1], self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)

        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # ---------------------------------------------------------#
        #   设置字体
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


# --------------------------------------------------#
#   单次卷积DarknetConv2D
#   正则化系数为5e-4
#   如果步长为2则自己设定padding方式。
# --------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)

    return x, y


# ---------------------------------------------------#
#   FPN网络的构建，并且获得预测结果
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # ---------------------------------------------------#
    #   生成darknet53的主干模型
    #   获得三个有效特征层，他们的shape分别是：
    #   52,52,256
    #   26,26,512
    #   13,13,1024
    # ---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # ---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    # ---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    # 13,13,512 -> 13,13,256 -> 26,26,256
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)

    # 26,26,256 + 26,26,512 -> 26,26,768
    x = Concatenate()([x, feat2])
    # ---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    # ---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x = Concatenate()([x, feat1])
    # ---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    # ---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # ---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    # ---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    # -----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    # -----------------------------------------------------------------#
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# ---------------------------------------------------#
#   获取每个box和它的得分
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    # -----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # -----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    # -----------------------------------------------------------------#
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # -----------------------------------------------------------------#
    #   获得最终得分和框的位置
    # -----------------------------------------------------------------#
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# ---------------------------------------------------#
#   图片预测
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    # ---------------------------------------------------#
    #   获得特征层的数量，有效特征层的数量为3
    # ---------------------------------------------------#
    num_layers = len(yolo_outputs)
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
    #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
    #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # -----------------------------------------------------------#
    #   这里获得的是输入图片的大小，一般是416x416
    # -----------------------------------------------------------#
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # -----------------------------------------------------------#
    #   对每个特征层进行处理
    # -----------------------------------------------------------#
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # -----------------------------------------------------------#
    #   将每个特征层的结果进行堆叠
    # -----------------------------------------------------------#
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # -----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    # -----------------------------------------------------------#
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # -----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        # -----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # -----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        # -----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # -----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是
        #   框的位置，得分与种类
        # -----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
