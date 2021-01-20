import numpy as np
import os
import glob
import csv
import cv2
import pandas as pd
import h5py
import xlwt
import xlrd
import xlutils
from xlrd import open_workbook
from xlutils.copy import copy
# from skimage import io
#训练数据路径
train_image_path="../tcdata/tile_round1_train_20201231/train_imgs"
patch_size = 416
data_path = "../user_data/data.h5"
list_train_img = []
list_bbox = []
list_category = []
# count 统计
count = 0
#设置index [name,category,bbox]
index = ['name','category','bbox']
def main():
    # Get data_path list_dir
    Dir = os.listdir(train_image_path)
    First = False
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
        # open excel to save category and bbox
        # data_frame  = open_workbook("../user_data/Temp data/train.xls")
        # row = data_frame.sheets()[0]#获取行数
        #对象转换
        # data_frame = copy(data_frame)
        # frame = data_frame.get_sheet(0)
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
            #相对坐标转换
            #相对中心点坐标,以patch为边界，左上为0(相对)
            X_zero_Temp = X_center - patch_size//2
            Y_zero_Temp = Y_center - patch_size//2
            #bbox
            Xmin_patch = Xmin - X_zero_Temp
            Ymin_patch = Ymin - Y_zero_Temp
            Xmax_patch = Xmax - X_zero_Temp
            Ymax_patch = Ymax - Y_zero_Temp
            list_img_patch = [Xmin_patch,Ymin_patch,Xmax_patch,Ymax_patch]

            # cut slice
            img_slice = im[X_center-patch_size//2:X_center+patch_size//2,Y_center-patch_size//2:Y_center+patch_size//2,:]
            # get category
            category = int(label_df["category"][i])
            # save img
            # path = os.path.join('../user_data/Temp data/train_img',img1+'_'+str(i)+'.bmp')
            if img_slice.size==0:
                continue
            cv2.imwrite(os.path.join('../user_data/Temp_data/train_img',img1+'_'+str(i)+'.bmp'),img_slice)
            # save bbox and category
            #trans into dict
            train_label = {'name':img1+'_'+str(i),'category':category,'bbox':list_img_patch}
            train_label1 = [img1+'_'+str(i),category,list_img_patch]
            # data_frame.append(train_label)
            # frame.write(row,0,img1+'_'+i)
            # frame.write(row,1,list_img)
            # frame.write(row,2,category)
            # row = row+1
            if First:
                with open("../user_data/Temp_data/train1.csv","a+") as f:
                    file_writer = csv.writer(f)
                    file_writer.writerow(train_label1)
                # file1_label['name'] = img1+'_'+str(i)
                # file1_label['category'] = category
                # file1_label['bbox'] = list_img
                # file1_label.write
                # file1_label.to_csv("../user_data/Temp_data/train.csv",index=False)
            else:
                # 创建csv
                file_label = pd.DataFrame.from_dict(data=train_label,orient='index').T
                file_label.to_csv("../user_data/Temp_data/train1.csv",mode='a',index=False)
                First = True


# def save_h5(list1,list2,list3):
#     """
#     :param list1: list_img
#     :param list2: list_bbox
#     :param list3: list_category
#     :return:
#     """
#     # trans into array
#     img_patch = np.asarray(list_train_img)
#     img_bbox = np.asarray(list_bbox)
#     img_category = np.asarray(list_category)
#     # dataset : img_patch img_bbox img_category
#     with h5py.File(data_path,'w') as hf:
#         hf.create_dataset("img_patch",data=img_patch)
#         hf.create_dataset("img_bbox",data=img_bbox)
#         hf.create_dataset("img_category",data=img_category)
#         hf.close()



#程序执行
if __name__ == '__main__':
    main()
    # save_h5(list_train_img, list_bbox, list_category)
    print("sucessful !")