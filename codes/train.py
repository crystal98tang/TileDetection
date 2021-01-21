import keras.backend as K
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from codes.core.config import cfg
from codes.core.utils import get_anchors, get_classes, get_data, get_random_data, read_csv
from codes.core.yolov3 import yolo_body
from codes.core.loss import yolo_loss


# ---------------------------------------------------#
#   数据生成器
# ---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:  # 训练完一个循环 再打乱
                np.random.shuffle(annotation_lines)
            # image, box = get_data(annotation_lines[i])
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# ---------------------------------------------------#
#   读入处理并输出y_true
# ---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    # assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors) // 3
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
    #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
    #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # -----------------------------------------------------------#
    #   获得框的坐标和图片的大小
    # -----------------------------------------------------------#
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # -----------------------------------------------------------#
    #   通过计算获得真实框的中心和宽高
    #   中心点(m,n,2) 宽高(m,n,2)
    # -----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # -----------------------------------------------------------#
    #   将真实框归一化到小数形式
    # -----------------------------------------------------------#
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # -----------------------------------------------------------#
    #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    # -----------------------------------------------------------#
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # -----------------------------------------------------------#
    #   [9,2] -> [1,9,2]
    # -----------------------------------------------------------#
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # -----------------------------------------------------------#
    #   长宽要大于0才有效
    # -----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # -----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        # -----------------------------------------------------------#
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # -----------------------------------------------------------#
        #   计算所有真实框和先验框的交并比
        #   intersect_area  [n,9]
        #   box_area        [n,1]
        #   anchor_area     [1,9]
        #   iou             [n,9]
        # -----------------------------------------------------------#
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # -----------------------------------------------------------#
        #   维度是[n,] 感谢 消尽不死鸟 的提醒
        # -----------------------------------------------------------#
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            # -----------------------------------------------------------#
            #   找到每个真实框所属的特征层
            # -----------------------------------------------------------#
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # -----------------------------------------------------------#
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    # -----------------------------------------------------------#
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # -----------------------------------------------------------#
                    #   k指的的当前这个特征点的第k个先验框
                    # -----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    # -----------------------------------------------------------#
                    #   c指的是当前这个真实框的种类
                    # -----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    # -----------------------------------------------------------#
                    #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                    #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                    #   1代表的是置信度、80代表的是种类
                    # -----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

if __name__ == "__main__":
    # 获取patch图片和标签
    annotation_path = cfg.PATH.annotation_path
    # 训练后的模型保存路径
    log_dir = cfg.PATH.logs
    # 权值文件
    weights_path = cfg.PATH.weight_path
    # 输入的shape大小
    input_shape = cfg.TRAIN.input_size
    # 是否对损失进行归一化
    normalize = True
    # 获取classes和anchor
    class_names = get_classes(cfg.PATH.classes_info)
    anchors = get_anchors(cfg.PATH.anchors_info)
    # 一共有多少类和多少先验框
    num_classes = len(class_names)
    num_anchors = len(anchors)

    K.clear_session()

    # ------------------------------------------------------#
    # 创建yolo模型
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    # ------------------------------------------------------#
    #   载入预训练权重
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    # ------------------------------------------------------#
    #   在这个地方设置损失，将网络的输出结果传入loss函数
    #   把整个模型的输出作为loss
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'normalize': normalize})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   提示OOM或者显存不足请调小Batch_size
    freeze_layers = 184
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
    # -------------------------------------------------------------------------------#
    #   训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # ----------------------------------------------------------------------#
    #   验证集的划分
    val_split = cfg.TRAIN.val_split
    lines = read_csv(cfg.PATH.annotation_path)
    np.random.seed(10101)
    np.random.shuffle(lines)
    #
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    # ------------------------------------------------------#
    if True:
        Init_epoch = 0
        Freeze_epoch = cfg.TRAIN.freeze_epoch
        batch_size = cfg.TRAIN.batch_size
        learning_rate_base = cfg.TRAIN.lr_init
        #
        model.compile(optimizer=Adam(lr=learning_rate_base), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})
        #
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           random=False),
            validation_steps=max(1, num_val // batch_size), epochs=Freeze_epoch,
            initial_epoch=Init_epoch, callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    for i in range(freeze_layers): model_body.layers[i].trainable = True

    # 解冻后训练
    if True:
        Freeze_epoch = cfg.TRAIN.freeze_epoch
        Epoch = cfg.TRAIN.epoch
        batch_size = cfg.TRAIN.batch_size // 2
        learning_rate_base = cfg.TRAIN.lr_normal
        #
        model.compile(optimizer=Adam(lr=learning_rate_base), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})
        #
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Epoch,
            initial_epoch=Freeze_epoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'last1.h5')
