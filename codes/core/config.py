from easydict import EasyDict as edict

__C = edict()
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
    "0": (0, 0, 0),
    "1": (255, 0, 0),
    "2": (0, 255, 0),
    "3": (0, 0, 255),
    "4": (128, 128, 0),
    "5": (0, 128, 128),
    "6": (128, 0, 128)
}

# Path config
__C.PATH = edict()
__C.PATH.origin_train_img_path = '../tcdata/tile_round1_train_20201231/train_imgs/'  # 原始训练集
__C.PATH.origin_train_anno_path = '../tcdata/tile_round1_train_20201231/train_annos.json/'  # 原始训练集
__C.PATH.origin_test_img_path = '../tcdata/tile_round1_testA_20201231/testA_imgs/' #测试数据集 "../user_data/Temp_data/train_img_mult_cutted_test/Images"#
#
__C.PATH.patch_path = '../user_data/Temp_data/train_img2'  # 训练集图片路径(旧/已弃)
__C.PATH.mult_patch_path = '../user_data/Temp_data/train_img_mult_cutted_fin'  # 裁剪结果存放目录
#
__C.PATH.logs = '../logs/'  # 训练后模型保存路径
__C.PATH.classes_info = '../user_data/model_data/classes.txt'  # 分类标签文件路径
__C.PATH.anchors_info = '../user_data/model_data/anchors.txt'  # 锚点文件路径
__C.PATH.weight_path = '../user_data/model_data/yolo_weights.h5'  # 预训练权值文件
#
__C.PATH.test_model_path = '../logs/100_last.h5'    # TODO:改这里取v3模型
cfg.PATH.test_model_v4_path = '../logs/100_last.h5' # TODO:改这里取v4模型
# __C.PATH.test_patch_patch = '../user_data/Temp_data/train_img_mult_cutted_test/'  # 测试集图片路径
# __C.PATH.temp_test_path = '../user_data/test/'  # 临时测试路径
# Train config
__C.TRAIN = edict()

__C.TRAIN.batch_size = 21
__C.TRAIN.input_size = (416, 416)
__C.TRAIN.val_split = 0.3  # 验证集/训练集 划分比例
__C.TRAIN.true_val = 0.1  # 验证集中实际使用比例
__C.TRAIN.lr_init = 1e-3
__C.TRAIN.lr_normal = 1e-4
#
__C.TRAIN.freeze_epoch = 1
__C.TRAIN.total_epoch = 2

# TEST config
__C.TEST = edict()
__C.TEST.batch_size = 5      # TODO： 进行resize 提升预测速度
__C.TEST.max_box_num = 20
__C.TEST.input_size = 416    # 输入网络的size
__C.TEST.patch_size = 416    # TODO：增大所切patch尺寸 进行resize 提升预测速度
__C.TEST.gap = 150
__C.TEST.score_threshold = 0.4
__C.TEST.iou_threshold = 0.2     # 全取 直接融合后nms
__C.TEST.visual_show = True
__C.TEST.out_result = True
