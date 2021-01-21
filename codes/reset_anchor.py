import glob
import xml.etree.ElementTree as ET
import tqdm
import numpy as np
import pandas as pd
from codes.core.utils import read_csv
from codes.core.config import cfg
from codes.core.kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "E:/pytorch/ultralytics-yolov3/yolov3-sheep/data/Annotations/"
#以正斜杠/这种形式可以防止反斜杠带来的转义错误
CLUSTERS = 9
image_ann = {}

def patch_load_dataset(lines):
   dataset = []
   for row in tqdm.tqdm(lines):
      height = width = 416
      bbox = list(eval(row[2]))
      xmin = int(float(bbox[0])) / width
      ymin = int(float(bbox[1])) / height
      xmax = int(float(bbox[2])) / width
      ymax = int(float(bbox[3])) / height
      dataset.append([xmax - xmin, ymax - ymin])
   return np.array(dataset)


def load_dataset(lines):
    dataset = []
    for name in tqdm.tqdm(image_ann.keys()):
        indexs = image_ann[name]
        height, width = indexs[0][3], indexs[0][4]
        for tup in indexs:
            bbox = tup[1]
            xmin = int(float(bbox[0])) / width
            ymin = int(float(bbox[1])) / height
            xmax = int(float(bbox[2])) / width
            ymax = int(float(bbox[3])) / height
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)

if __name__ == '__main__':
    """
    使用Patch数据
    Accuracy: 77.28%
    Boxes:
    [34. 10. 82. 13. 18. 10.  8. 13.  7.]-[29. 12. 89. 14. 17.  9.  8. 11.  8.]
    ###result: 7,8,  8,8,  10,9,  10,11,  13,12,  13,14,  18,17,  34,29,  82,89
    Ratios:
    [0.83, 0.87, 0.92, 0.93, 1.0, 1.06, 1.11, 1.17, 1.18]
    """
    lines = read_csv(cfg.PATH.annotation_path)
    data = patch_load_dataset(lines)
    while True:
        out = kmeans(data, k=CLUSTERS)
        print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
        print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))
        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
        print("Ratios:\n {}".format(sorted(ratios)))

    """
    使用原始数据
    Accuracy: 74.23%
    Boxes:
     [5.28125    0.86328125 1.21875    0.8125     0.55859375 2.4375
     0.7109375  0.40625    0.609375  ]-[7.28       1.10933333 1.456      0.832      0.90133333 2.25828571
     1.04       0.55466667 0.76266667]
    Ratios:
     [0.62, 0.68, 0.73, 0.73, 0.78, 0.8, 0.84, 0.98, 1.08]
    """
    # # fixme: anchor过小
    # df = pd.read_json("../tcdata/tile_round1_train_20201231/train_annos.json", orient='records')
    # image_ann = {}
    # for tup in df.itertuples():
    #     name = tup[0]
    #     if name not in image_ann:
    #         image_ann[name] = []
    #     image_ann[name].append(tup)
    # while True:
    #     data = load_dataset(image_ann)
    #     out = kmeans(data, k=CLUSTERS)
    #     print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    #     print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))
    #     ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    #     print("Ratios:\n {}".format(sorted(ratios)))

    print("successful!")
