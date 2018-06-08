# -*- coding=utf-8 -*-
##############################################
# 将训练集分成两部分：1）训练用；2）验证用
# 分布规律：
#     1）对于每个label训练集占p，验证集占1-p
#     2）打乱数据
# 输入：
#     train.txt的路径
# 输出：
#     split_train.txt
#     split_valid.txt
# 输出格式：
#     每行包括图片绝对路径和label，形如：
#     /home/maozezhong/Desktop/baidu_dianshi/datasets/train/7c1ed21b0ef41bd5bd0864e95dda81cb39db3d57.jpg 41
##############################################

from random import shuffle

rate = 0.1

root_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train/'
path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train.txt'
train_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/split_train.txt'
valid_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/split_valid.txt'
train_data = [] #放所有用于训练的图片数据
valid_data = [] #放所有用于验证的图片数据
label_path = {} #key为label，value为label为该label的所有path
with open(path, 'r') as f:

    #1#得到每个标签对应的所有图片路径
    for line in f.readlines():
        label = line.split(' ')[1]
        if label in label_path.keys():
            label_path[label].append(line)
        else:
            label_path[label] = [line]
    
    #2#对于每个标签，打乱其对应的所有图片路径，并取前1-rate放入train_data,后rate放入valid_data
    for key in label_path.keys():
        pic_paths = label_path[key]
        shuffle(pic_paths)
        length = len(pic_paths)
        train_length = int((1-rate)*length) 
        for i in range(train_length):
            train_data.append(pic_paths[i])
        for i in range(train_length, length):
            valid_data.append(pic_paths[i])
    
    #3#打乱train_data以及valid_data的顺序，并写入目标文件
    shuffle(train_data)
    shuffle(valid_data)
    train_file = open(train_path, 'w')
    valid_file = open(valid_path, 'w')
    for data in train_data:
        train_file.write(root_path+data)
    for data in valid_data:
        valid_file.write(root_path+data)
    train_file.close()
    valid_file.close()