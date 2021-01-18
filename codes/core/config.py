from easydict import EasyDict as edict

__C                         = edict()
# Get config by: from config import cfg

cfg = __C
# Path

# Dict
__C.class_name_dic          = {
    "0": "background",
    "1": "edge error",
    "2": "angle error",
    "3": "White spot defect",
    "4": "Light color blemish",
    "5": "Dark dot block defect",
    "6": "Aperture flaws"
}
__C.lable_color             = {
    "0": (0,0,0),
    "1": (255,0,0),
    "2": (0,255,0),
    "3": (0,0,255),
    "4": (128,128,0),
    "5": (0,128,128),
    "6": (128,0,128)
}

