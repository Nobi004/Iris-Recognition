import pandas as pd
import shutil
import os

train = pd.read_csv("D:\\Dataset\\iris\\csv\\train.csv")
test = pd.read_csv("D:\\Dataset\\iris\\csv\\test.csv")
print(train,test)
train_img_list = train['img']
train_label_list = train['label']
test_img_list = test['img']
test_label_list = test['label']

train_size = len(train_img_list)
test_size = len(test_img_list)

copy_dir1="D:\\Dataset\\iris\\process_data\\train"
copy_dir2="D:\\Dataset\\iris\\process_data\\test"

for i in range(train_size):
    train_path = train_img_list[i]
    print(train_path)
    img_name = train_path.split('\\')[-1]
    copy_path1 = os.path.join(copy_dir1,img_name)
    print(train_path,copy_path1)
    shutil.copy(train_path,copy_path1)

for i in range(test_size):
    test_path = test_img_list[i]
    print(test_path)
    img_name = test_path.split('\\')[-1]
    copy_path2 = os.path.join(copy_dir2,img_name)
    shutil.copy(test_path,copy_path2)

