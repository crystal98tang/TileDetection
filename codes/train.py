import keras.backend as K
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from codes.core.config import cfg
from codes.core.utils import get_anchors, get_classes, get_random_data
from codes.core.yolov3 import yolo_body

#---------------------------------------------------#
#   数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:    # 训练完一个循环 再打乱
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

#---------------------------------------------------#
#   读入处理并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    # TODO: 处理标签

    return y_true

if __name__ == "__main__":
    # 获取patch图片和标签 TODO:修改路径
    annotation_path = '.h5'
    # 训练后的模型保存路径
    log_dir = '../logs/'
    # 权值文件
    weights_path = cfg.TRAIN.weight
    # 输入的shape大小
    input_shape = cfg.TRAIN.input_size
    # 获取classes和anchor
    class_names = get_classes(cfg.TRAIN.classes)
    anchors = get_anchors(cfg.TRAIN.anchors)
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

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model = Model([model_body.input, *y_true])

    # -------------------------------------------------------------------------------#
    #   训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # ----------------------------------------------------------------------#
    #   验证集的划分
    #   验证集和训练集的比例为1:9
    val_split = 0.1
    with open(annotation_path) as f:    #TODO：修改为h5读取方式
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    # ------------------------------------------------------#
    if True:
        epoch = cfg.TRAIN.epoch
        batch_size = cfg.TRAIN.batch_size
        learning_rate_base = cfg.TRAIN.lr_init
        # Complie
        model.compile(optimizer=Adam(lr=learning_rate_base), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})
        # Fit_generator
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights.h5')
