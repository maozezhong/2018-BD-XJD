# -*- coding=utf-8 -*-
# 可视化数据集每类图片

import os
import cv2

label = 1
path = os.getcwd() + '/datasets/train.txt'
with open(path, 'r') as f:
    for line in f.readlines():
        pic_name = os.getcwd()+'/datasets/train/'+line.split(' ')[0]
        cur_label = int(line.split(' ')[1])
        if cur_label == label:
            print(label)
            img = cv2.imread(pic_name)
            cv2.imshow('data',img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            label = label + 1