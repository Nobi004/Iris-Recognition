import numpy as np
import cv2
from script import IrisLocalization ,IrisNormalization,ImageEnhancement
from Gabor import FeatureExtraction
import datetime
import pandas as pd

train = pd.read_csv("D:\\Dataset\\iris\\csv\\train.csv")
test = pd.read_csv("D:\\Dataset\\iris\\csv\\test.csv")

train_image_list = train['img']
train_label_list = train['label']
test_image_list = test['img']
test_label_list = test['label']

train_size = len(train_image_list)
test_size = len(test_image_list)

train_features = np.zeros((train_size,1536))
train_classes = np.zeros(train_size,dtype= np.uint8)
test_features = np.zeros((test_size,1536))
test_classes = np.zeros(test_size,dtype= np.uint8)

starttime = datetime.datetime.now()



for j in range(test_size):
    test_path = test_image_list[j]
    img_name = test_path.split('\\')[-1].split('.')[0]
    img = cv2.imread(test_path,0)
    iris,pupil= IrisLocalization(img)

    normalized = IrisNormalization(img,pupil,iris)
    cv2.imwrite("D:\\Dataset\\iris\\process_data\\{}.jpg".format(img_name), normalized)
    print('test_features:',test_image_list[j],test_label_list[j],'{}/{}'.format(j,test_size))

endtime = datetime.datetime.now()
