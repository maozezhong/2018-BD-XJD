# -*- coding=utf-8 -*-
############################
# 线下对图片先进行一个剪切
############################

from keras.preprocessing.image import img_to_array
from PIL import Image
import os
import shutil
import cv2

def show_pic(img):
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

index = 0
# 图片的原始路径
path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train_for_flow'
for parent, _, files in os.walk(path):
    
    # 先删除之前扩充的
    for file in files:
        pic_path = parent + '/' + file
        if file[-10:] == '_noise.jpg' or file[-10:] == '_light.jpg' or file[-18:] == '_noiseAndlight.jpg':
            os.remove(pic_path)
    
    for file in files:

        index = index + 1
        print(index)

        if file[-10:] == '_noise.jpg' or file[-10:] == '_light.jpg' or file[-18:] == '_noiseAndlight.jpg':
            continue

        ori_pic_path = parent + '/' + file
        im = Image.open(ori_pic_path)
        img_w, img_h = im.size

        # 裁剪以图片中心开始的一个宽为w，高为h的框
        p = 0.9 #裁剪的概率
        w = p * img_w
        h = p * img_h
        x = (img_w - w) / 2
        y = (img_h - h) / 2
        

        region = im.crop((x, y, x+w, y+h))
        # 可视化
        # show_pic(img_to_array(region))

        # 裁剪后图片的存储位置
        target_pic_root = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train_for_flow_crop/'+ parent.split('/')[-1]
        if not os.path.exists(target_pic_root):
            os.mkdir(target_pic_root)
        target_pic_path = target_pic_root + '/' + file
        region.save(target_pic_path)