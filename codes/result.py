import json

final_path = '/home/aistudio/PaddleDetection/final_results.json'
results_path = '/home/aistudio/PaddleDetection/results.json'


def result(final_path, results_path):
    with open(results_path, "r") as f:
        test_result = json.load(f)
    # 获取图片id对应的图片名字字典
    imgs = test_result["images"]
    dict_img = {}
    for img in imgs:
        img_name = img["file_name"]
        img_id = img["id"]
        dict_img[str(img_id)] = img_name
    # print(dict_img["1"])
    # 按照提交格式对应字段
    final_results = []
    annotations = test_result["annotations"]
    for ann in annotations:
        dict_ann = {}
        # 设置图片name
        # 将图片id对应为name
        ann_name_id = str(ann["image_id"])
        dict_ann["name"] = dict_img[ann_name_id]
        # 设置类别category
        dict_ann["category"] = ann["category_id"]
        # 设置bbox
        # 之前预测的bbox中格式为【左上角横坐标x，左上角纵坐标y，框的高h，框的宽w】
        # 提交格式要求的bbox格式为【左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标】
        bbox = ann["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        dict_ann["bbox"] = bbox
        # 设置置信度score
        dict_ann["score"] = ann["score"]
        final_results.append(dict_ann)

    # print(final_results[0])
    json.dump(final_results, open(final_path, 'w'), indent=4)


if __name__ == '__main__':

    result()