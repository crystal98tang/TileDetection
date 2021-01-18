import numpy as np
import os
import glob
import cv2
import pandas as pd
import h5py
import ast

#训练数据路径
train_image_path="../tcdata/tile_round1_train_20201231/train_imgs"
patch_size = 512
count = 0
data_path = "../user_data/data.h5"
list_train_img = []
list_bbox = []
list_category = []
def main():
    # Get data_path list_dir
    Dir = os.listdir(train_image_path)
    for img in Dir:
        # get img_name
        img1 = os.path.splitext(img)[0]
        im = cv2.imread(os.path.join(train_image_path, img))
        xls_path = os.path.join("../user_data/label_xls",str(img1)+".xls")
        # read dataframe about img_xls
        label_df = pd.read_excel(xls_path)
        # print(label_df['bbox'][0])
        # get Length about label point
        num = len(label_df['bbox'])
        # get slice by label_center
        print(img1)
        for i in range(num):
            # label_df['bbox'] =  pd.to_numeric(label_df['bbox'])   #"[1,2,3,4]
            list_img = list(eval(label_df['bbox'][i]))
            # list_img [Xmin Ymin Xmax Ymax]
            # get value
            Xmin = int(list_img[0])
            Ymin = int(list_img[1])
            Xmax = int(list_img[2])
            Ymax = int(list_img[3])
            #deal with X
            X_center = (Xmin + Xmax)//2
            #deal with Y
            Y_center = (Ymin + Ymax)//2
            # cut slice
            img_slice = im[X_center-patch_size//2:X_center+patch_size//2,Y_center-patch_size//2:Y_center+patch_size//2,:]
            # get category
            category = int(label_df["category"][i])
            # save list
            list_train_img.append(img_slice)
            list_bbox.append(list_img)
            list_category.append(category)

def save_h5(list1,list2,list3):
    """
    :param list1: list_img
    :param list2: list_bbox
    :param list3: list_category
    :return:
    """
    # trans into array
    img_patch = np.asarray(list_train_img)
    img_bbox = np.asarray(list_bbox)
    img_category = np.asarray(list_category)
    # dataset : img_patch img_bbox img_category
    with h5py.File(data_path,'w') as hf:
        hf.create_dataset("img_patch",data=img_patch)
        hf.create_dataset("img_bbox",data=img_bbox)
        hf.create_dataset("img_category",data=img_category)
        hf.close()



#程序执行
if __name__ == '__main__':
    main()
    save_h5(list_train_img, list_bbox, list_category)
    print("sucessful !")