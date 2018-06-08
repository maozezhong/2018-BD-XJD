# -*- coding=utf-8 -*-
######################################################
# 一些辅助函数
######################################################
import numpy as np

import imageio
from skimage.transform import resize as imresize

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

IN_SIZE = 448

# 获得索引对应的真实label,返回一个字典，key为索引，value为label
def get_index_class():
    classes = []
    for i in range(1,101):
        classes.append(str(i))
    classes = sorted(classes)
    return dict(zip(range(100), classes))
    # return dict(zip(classes, range(100)))

# 获得每个label对应的名称，返回的是一个字典
def get_label_name(path):
    res = dict()
    with open(path, 'r') as f:
        for line in f.readlines():
            label_name = line.split('.')
            label = label_name[0]
            name = label_name[1]
            res[label] = name
    return res

# rescale图片
def rescale(x):
    return x / 255.

# resize图片大小 
def img_reshape(img):
    img = imresize(img, (331, 331, 3))
    return img

# 统计每一类的样本张数
def count_picNum_per_label(path):
    '''
    输入：
        path：train.txt的文件路径
    '''

    count = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            label = line.split(' ')[1]
            if label in count.keys():
                count[label] = count[label]+1
            else:
                count[label] = 1
    
    print(count)

################ keras使用迭代器载入数据 ################
def process_line(line):  
    '''
    输入：
        line：split_train.txt的一行数据，形如：
        “/home/maozezhong/Desktop/baidu_dianshi/datasets/train/2fdda3cc7cd98d1021c7771c2d3fb80e7bec908f.jpg 62”
        包括图片的绝对路径和标签，以空格分隔
    输出：
        x:图片的数据矩阵
        y:图片对应的label的ont-hot形式
    '''
    if len(line.split('/'))==1:
        root = '/home/maozezhong/Desktop/baidu_dianshi/datasets/test'
        pic_path = root + '/' + line.split(' ')[0]
    else:
        pic_path = line.split(' ')[0]
    img = load_img(pic_path, target_size=(IN_SIZE, IN_SIZE, 3))
    x = img_to_array(img)
    # x = preprocess_input(x)
    x = rescale(x)

    index_label = get_index_class()
    label_index = {value:key for key,value in index_label.items()}
    label = line.split(' ')[1].strip()
    index = label_index[label]

    y = np.zeros(100)   #100类，转成one-hot的格式
    y[index] = 1

    return x,y  

def generate_arrays_from_file(path, batch_size):  
    '''
    输入：
        path：split_train.txt的文件路径
        batch_size：每个批次要返回的图片张数
    输出：
        batch_size张图片的矩阵信息及label
    '''
    while 1:  
        f = open(path)  
        cnt = 0  
        X =[]  
        Y =[]  
        for line in f.readlines():  
            # create Numpy arrays of input data  
            # and labels, from each line in the file  
            x, y = process_line(line)  
            X.append(x)  
            Y.append(y)  
            cnt += 1  
            if cnt==batch_size:  
                cnt = 0  
                yield (np.array(X), np.array(Y))  
                X = []  
                Y = []  
    f.close()
 

################keras使用迭代器载入数据end################

# 得到训练数据的下一个batch的数据
def next_batch(path, batch_size):  
    '''
    输入：
        path：split_train.txt的文件路径
        batch_size：每个批次要返回的图片张数
    输出：
        batch_size张图片的矩阵信息及label
    '''
 
    f = open(path)  
    cnt = 0  
    X =[]   
    Y =[]
    for line in f.readlines():  
        # create Numpy arrays of input data  
        # and labels, from each line in the file  
        x, y = process_line(line)
        X.append(x)  
        Y.append(y)
        cnt += 1  
        if cnt%batch_size==0:  
            yield (np.array(X),np.array(Y))
            X = []  
            Y = []
        elif cnt==len(f.readlines()):
            yield (np.array(X),np.array(Y)) 
            X = [] 
            Y = [] 
    f.close()  

if __name__ == '__main__':
    path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/split_train.txt'
    batch_size = 32
    for i in generate_arrays_from_file('/home/maozezhong/Desktop/baidu_dianshi/result/truth.txt', 10):
        print(i)