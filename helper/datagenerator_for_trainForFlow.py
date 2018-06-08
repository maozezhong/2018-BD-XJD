# -*- coding=utf-8 -*-
###############################
# 为train_for_flow增强数据，
# 目前打算对每一张图片加上3张图片，一张加噪声，一张加改亮度，一张加噪声和亮度叠加
###############################
import os
import cv2
import random
from skimage.util import random_noise
from skimage import exposure

# 加高斯噪声
def addNoise(img):
    '''
    注意：输出的像素是[0,1]之间,所以乘以5得到[0,255]之间
    '''
    return random_noise(img, mode='gaussian', seed=13, clip=True)*255

def changeLight(img):
    rate = random.uniform(0.5, 1.5)
    # print(rate)
    img = exposure.adjust_gamma(img, rate) #大于1为调暗，小于1为调亮;1.05
    return img

def show_pic(img):
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def datagenerate():
    path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train_for_flow_crop'
    for i in range(1,101):
        print('generate data for label : '+str(i))
        root_path = path + '/' + str(i)
        for _, _, files in os.walk(root_path):

            # 先删除之前生成的
            for file in files:
                pic_path = root_path + '/' + file
                if file[-10:] == '_noise.jpg' or file[-10:] == '_light.jpg' or file[-18:] == '_noiseAndlight.jpg':
                    os.remove(pic_path)

            for file in files:

                pic_path = root_path + '/' + file

                # 如果遇到之前生成的pic的文件名，则跳过（因为前面虽然已经删除了之前生成的图片，但是files还是包括这些名称的）
                if file[-10:] == '_noise.jpg' or file[-10:] == '_light.jpg' or file[-18:] == '_noiseAndlight.jpg':
                    continue
                    # print(file)
                
                img = cv2.imread(pic_path)

                img1 = addNoise(img)
                pic1_path = pic_path[:-4]+'_noise.jpg'
                cv2.imwrite(pic1_path, img1)

                img2 = changeLight(img)
                pic2_path = pic_path[:-4] + '_light.jpg'
                cv2.imwrite(pic2_path, img2)

                img3 = changeLight(img)
                img3 = addNoise(img3)
                pic3_path = pic_path[:-4]+'_noiseAndlight.jpg'
                cv2.imwrite(pic3_path, img3)

                # show_pic(img)
                # show_pic(img1)
                # show_pic(img2)
                # show_pic(img3)

if __name__ == '__main__':
    datagenerate()
