import numpy as np
import os
import glob
import csv
import cv2
import pandas as pd
import h5py
import xlwt
# from skimage import io
#训练数据路径
train_image_path="../tcdata/tile_round1_train_20201231/train_imgs"
patch_size = 416
data_path = "../user_data/data.h5"
list_train_img = []
list_bbox = []
list_category = []
img_path = '../user_data/Temp_data/train_img2'
csv_path = "../user_data/Temp_data/train2.csv"
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
        # get Length about label point
        num = len(label_df['bbox'])
        # get slice by label_center
        print(img1)
        for i in range(num):
            # label_df['bbox'] =  pd.to_numeric(label_df['bbox'])   #"[1,2,3,4]
            list_img = list(eval(label_df['bbox'][i]))
            # list_img [Xmin Ymin Xmax Ymax]
            # get value
            Xmin = float(list_img[0])
            Ymin = float(list_img[1])
            Xmax = float(list_img[2])
            Ymax = float(list_img[3])
            #deal with X
            X_center = (Xmin + Xmax)//2
            #deal with Y
            Y_center = (Ymin + Ymax)//2
            #相对坐标转换
            #相对中心点坐标,以patch为边界，左上为0(相对),XY上两个基点
            X_zero_Temp = X_center - patch_size//2
            Y_zero_Temp = Y_center - patch_size//2
            #bbox
            Xmin_patch = Xmin - X_zero_Temp
            Ymin_patch = Ymin - Y_zero_Temp
            Xmax_patch = Xmax - X_zero_Temp
            Ymax_patch = Ymax - Y_zero_Temp
            list_img_patch = [Xmin_patch,Ymin_patch,Xmax_patch,Ymax_patch]

            # cut slice
            img_slice = im[int(Y_center-patch_size//2):int(Y_center+patch_size//2),int(X_center-patch_size//2):int(X_center+patch_size//2),:]
            # get category
            category = int(label_df["category"][i])
            # save img
            # path = os.path.join('../user_data/Temp data/train_img',img1+'_'+str(i)+'.bmp')
            if img_slice.size==0:
                continue
            cv2.imwrite(os.path.join(img_path,img1+'_'+str(i)+'.bmp'),img_slice)
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
                with open(csv_path,"a+") as f:
                    file_writer = csv.writer(f)
                    file_writer.writerow(train_label1)
            else:
                # 创建csv
                file_label = pd.DataFrame.from_dict(data=train_label,orient='index').T
                file_label.to_csv(csv_path,mode='a',index=False)
                First = True



#程序执行
if __name__ == '__main__':
    main()
    # save_h5(list_train_img, list_bbox, list_category)
    print("sucessful !")