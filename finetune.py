# coding=utf-8
#################################################
# finetune
#################################################
import pandas as pd
import numpy as np
import os
import imageio
import cv2
from helper.datagenerator_for_trainForFlow import show_pic
from helper.helper import get_label_name

from PIL import Image as pil_image
from skimage.transform import resize as imresize

from helper.helper import next_batch, generate_arrays_from_file, get_index_class
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import initializers
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Maximum
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from keras.applications.densenet import DenseNet169
#######################################
# 在训练的时候置为1
from keras import backend as K
K.set_learning_phase(1)
#######################################

##########指定GPU并且限制GPU用量##########
# import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

# # 指定第一块GPU可用 ne 93
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.ConfigProto()  
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)

# KTF.set_session(sess)
########################################

BATCH_SIZE = 4#32
EPOCHS = 350
RANDOM_STATE = 2018
IN_WIDTH, INT_HEIGHT = 448,448
learning_rate = 0.003
FREZZE_LAYER = -20

# 预先训练权值路径
weights_path = '/home/maozezhong/Desktop/baidu_dianshi/Model/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
TRAIN_DIR = '/home/maozezhong/Desktop/baidu_dianshi/datasets/train_for_flow_crop/'
VALID_DIR = '/home/maozezhong/Desktop/baidu_dianshi/datasets/valid_for_flow/'

def get_callbacks(filepath, patience=2):
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=1e-5, patience=patience, verbose=1, min_lr = 0.0000001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #该回调函数将在每个epoch后保存模型到filepath
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience+3, verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]

def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x) 
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
        model: keras model
    """
    for layer in model.layers[:FREZZE_LAYER]:
        layer.trainable = False
    for layer in model.layers[FREZZE_LAYER:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# focal loss with multi label
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor为softmax后的预测输出tensor，维度应该为[None, 100]
        target_tensor为目标tensor，维度应该与prediction_tensor一致
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops

        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
        entry_cross_ent = - alpha * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))
        return tf.reduce_sum(entry_cross_ent)
    return focal_loss_fixed

def add_attention(base_model):
    '''
    输入:
        base_model:基础模型
    输出:
        加了attention后的basemodel
    '''
    ################## add attention by mao 2019-6-5 21:00 ##################
    # 自己定义一个softMaxAxis，表示指定维度axis进行softmax
    # from keras.activations import softmax
    # def softMaxAxis(axis):
    #     def soft(x):
    #         return softmax(x, axis=axis)
    #     return soft
    
    # #加attention--对feature map加
    # from keras.layers import merge, Reshape, RepeatVector, Permute
    # x = base_model.output
    # x = Reshape((1, 49, 2048))(x)   #[None, 7, 7, 2048] -> [None, 1, 49, 2048]
    # x = Conv2D(1, (1,1), activation=softMaxAxis(-2), strides=(1,1), name='attention_feature')(x)    #[None, 1, 49, 2048] -> [None, 1, 49, 1]
    # x = Reshape((49,))(x)           #[None, 1, 49, 1] -> [None, 49]
    # x = RepeatVector(2048)(x)       #[None, 49] -> [None, 2048, 49]
    # x = Permute((2,1))(x)           #[None, 2048, 49] -> [None, 49, 2048]
    # x = Reshape((7, 7, 2048))(x)    #[None, 49, 2048] -> [None, 7, 7, 2048]
    # x = merge([base_model.output, x], name='attention_mul', mode='mul') #点乘
    # base_model = Model(inputs=base_model.input, outputs=x)

    #加attention--修改, 点乘后求和
    def getSum(input_tensor):
        '''
        input_tensor : [None, 49, 2048]
        Note:
            函数里面要用的都得在函数内import！！！！！！
        '''
        import keras.backend as K
        res = K.sum(input_tensor, axis=-2)
        return res

    from keras.layers import merge, Reshape, RepeatVector, Permute, Lambda
    x = base_model.output
    x = Reshape((1, 49, 2048))(x)   #[None, 7, 7, 2048] -> [None, 1, 49, 2048]
    x = Conv2D(1, (1,1), activation=softMaxAxis(-2), strides=(1,1), name='attention_feature')(x)    #[None, 1, 49, 2048] -> [None, 1, 49, 1]
    x = Reshape((49,))(x)           #[None, 1, 49, 1] -> [None, 49]
    x = RepeatVector(2048)(x)       #[None, 49] -> [None, 2048, 49]
    x = Permute((2,1))(x)           #[None, 2048, 49] -> [None, 49, 2048]
    x = Reshape((7, 7, 2048))(x)    #[None, 49, 2048] -> [None, 7, 7, 2048]
    x = merge([base_model.output, x], name='attention_mul', mode='mul') #点乘
    x = Reshape((49, 2048))(x)      #[None, 7, 7, 2048] -> [None, 49, 2048]
    x = Lambda(getSum)(x)           #对[49]这个位置求和并且reshape输出为[None, 2048]
    x = Reshape((1,1,2048))(x)      #[None, 2048] -> [None, 1, 1, 2048]
    base_model = Model(inputs=base_model.input, outputs=x)
    ############################## end of attention #########################

    return base_model

def get_model():
    '''
    获得模型
    '''
    # base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IN_WIDTH,INT_HEIGHT, 3))
    base_model = DenseNet169(include_top=False, weights=weights_path, input_shape=(IN_WIDTH,INT_HEIGHT, 3))

    # # add attention
    # base_model = add_attention(base_model)
    
    model = add_new_last_layer(base_model, 100)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[focal_loss()], metrics=['accuracy'])   #使用focal loss
    # model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def train_model():

    callbacks = get_callbacks(filepath='/home/maozezhong/Desktop/baidu_dianshi/datasets/weights/model_weight.hdf5', patience=3)
    model = get_model()
    # model.load_weights('/home/maozezhong/Desktop/baidu_dianshi/datasets/weights/model_weight.hdf5')

    # 数据增强，flow_from_dict()
    train_datagen = ImageDataGenerator(
        # preprocessing_function = preprocess_input,  #res50的预处理, finetune resnet50的时候使用
        rescale=1. / 255, 
        rotation_range = 30,    #旋转
        width_shift_range = 0.2,    #左右平移
        height_shift_range = 0.2,   #上下平移
        zoom_range = 0.3,           #放大缩小s
        shear_range = 0.3          #倾斜
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_DIR, 
        target_size = (IN_WIDTH, INT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=2018
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory = VALID_DIR,
        target_size = (IN_WIDTH, INT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=2018
    )

    model.fit_generator(
        train_generator, 
        steps_per_epoch = 1*(train_generator.samples // BATCH_SIZE + 1), #10 * TRAIN_PIC_NUM // BATCH_SIZE + 1, #加1防止数据丢失，trick    len(train_generator)+1, 
        epochs = EPOCHS,
        max_queue_size = 1000,
        workers = 1,
        verbose = 1,
        validation_data = valid_generator, #valid_generator,
        validation_steps = valid_generator.samples // BATCH_SIZE, #valid_generator.samples // BATCH_SIZE + 1, #len(valid_datagen)+1, 
        callbacks = callbacks
        )

def predict():
    '''
    对测试数据进行预测
    '''

    K.set_learning_phase(0)

    weights_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/weights/model1.hdf5'
    test_txt_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/test.txt'
    test_pic_root_path = '/home/maozezhong/Desktop/baidu_dianshi/datasets/test'

    res_path = '/home/maozezhong/Desktop/baidu_dianshi/result/res.csv'

    pic_names = []
    labels = []
    label_name = get_label_name('/home/maozezhong/Desktop/baidu_dianshi/datasets/label_name_reference.txt') #得到label对应的名称的dict
    index_class = get_index_class() #得到在flow_from_dicretory情况下的索引对应的label，calsses是按照字母序排序的

    #1#得到模型
    model = get_model()
    model.load_weights(weights_path)

    model_2 = get_model()
    model_2.load_weights('/home/maozezhong/Desktop/baidu_dianshi/datasets/weights/model2.hdf5')

    model_3 = get_model()
    model_3.load_weights('/home/maozezhong/Desktop/baidu_dianshi/datasets/weights/model3.hdf5')
    
    #2#预测
    num = 0
    with open(test_txt_path, 'r') as f:
        for line in f.readlines():
            num = num+1
            pic_path = test_pic_root_path + '/' + line.strip()

            ori_img = cv2.imread(pic_path)

            img = load_img(pic_path, target_size=(INT_HEIGHT, IN_WIDTH, 3))
            img = img_to_array(img)
            # img = preprocess_input(img)
            from helper.helper import rescale
            img = rescale(img)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0] + model_2.predict(img)[0] + model_3.predict(img)[0]
            index = list(prediction).index(np.max(prediction))
            label = index_class[index]

            #可视化一下
            print('第'+str(num)+'张标签是： '+str(label) + '  对应名称为： '+label_name[str(label)])
            show_pic(ori_img)

            #存入list
            pic_names.append(line.strip())
            labels.append(label)
            # print(line.strip())
            # print(label)
    
    #3#写入csv
    column = ['pic', 'label']
    dataframe = pd.DataFrame({'pic': pic_names, 'label': labels})
    dataframe.to_csv(res_path, index=False, header=False, sep=' ',columns=column)

def main():
    # train_model()
    predict()

if __name__=='__main__':
    main()
