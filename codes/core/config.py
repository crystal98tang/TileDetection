from easydict import EasyDict as edict

__C                             = edict()
# Get config by: from config import cfg

cfg = __C
# Dict
__C.class_name_dic = {
    "0": "background",
    "1": "edge error",
    "2": "angle error",
    "3": "White spot defect",
    "4": "Light color blemish",
    "5": "Dark dot block defect",
    "6": "Aperture flaws"
}
__C.lable_color = {
    "0": (0,0,0),
    "1": (255,0,0),
    "2": (0,255,0),
    "3": (0,0,255),
    "4": (128,128,0),
    "5": (0,128,128),
    "6": (128,0,128)
}

# Path
__C.PATH = edict()
__C.PATH.origin_train_img_path = '../tcdata/tile_round1_train_20201231/train_imgs/'   # 原始训练集
__C.PATH.origin_train_anno_path = '../tcdata/tile_round1_train_20201231/train_annos.json/'   # 原始训练集
#
__C.PATH.annotation_path = '../user_data/Temp_data/train1.csv'   # 训练集索引文件路径
__C.PATH.logs = '../logs/'  # 训练后模型保存路径
__C.PATH.classes_info = '../user_data/model_data/classes.txt'   # 分类标签文件路径
__C.PATH.anchors_info = '../user_data/model_data/anchors.txt'   # 锚点文件路径
__C.PATH.weight_path = '../user_data/model_data/yolo_weights.h5'  # 预训练权值文件
__C.PATH.patch_path = '../user_data/Temp_data/train_img'   # 训练集图片路径
# Train
__C.TRAIN = edict()

__C.TRAIN.batch_size = 10
__C.TRAIN.input_size = (416, 416)
__C.TRAIN.val_split = 0.1   # 训练集和验证集划分比例
__C.TRAIN.lr_init = 1e-3
__C.TRAIN.lr_normal = 1e-4

__C.TRAIN.freeze_epoch = 10
__C.TRAIN.epoch = 20
