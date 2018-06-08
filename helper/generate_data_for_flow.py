# -*- coding=utf-8 -*-
#############################################
# 产生适用keras,image_data_generator.flow_from_directory()输入形式的训练数据
# 在train_for_flow文件夹下生成1~100的子文件夹，并把对应lable的图片复制进去
#############################################

import os
import shutil
from datagenerator_for_trainForFlow import datagenerate

def generate_dir_for_flow(path, target_root_path):
    if os.path.exists(target_root_path):
        shutil.rmtree(target_root_path)
    with open(path, 'r') as f:
        for line in f.readlines():
            pic_path = line.split(' ')[0]
            label = line.split(' ')[1].strip()
            target_dic = target_root_path + '/' + label
            
            if not os.path.exists(target_dic):
                os.makedirs(target_dic)

            target_pic_path = target_dic + '/' + pic_path.split('/')[-1]
            print(pic_path + ' -> ' + target_pic_path)
            shutil.copyfile(pic_path, target_pic_path)

if __name__ == '__main__':
    split_train_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/split_train.txt'
    train_for_flow_root_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train_for_flow'
    split_valid_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/split_valid.txt'
    valid_for_flow_root_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/valid_for_flow'

    generate_dir_for_flow(split_train_path, train_for_flow_root_path)
    generate_dir_for_flow(split_valid_path, valid_for_flow_root_path)