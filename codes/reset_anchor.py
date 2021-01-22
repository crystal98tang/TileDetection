import tqdm
import numpy as np
from codes.core.utils import read_csv
from codes.core.config import cfg
from codes.core.kmeans import kmeans, avg_iou

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
    max_acc = 0
    while True:
        out = kmeans(data, k=CLUSTERS)
        acc = avg_iou(data, out) * 100
        if acc >= max_acc:
            max_acc = acc
            print("Accuracy: {:.2f}%".format(acc))
            x, y = out[:, 0] * 416, out[:, 1] * 416
            print("Boxes:\n {}-{}".format(np.sort(x), np.sort(y)))
            ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
            print("Ratios:\n {}".format(sorted(ratios)))
        else:
            print(".", end='')
