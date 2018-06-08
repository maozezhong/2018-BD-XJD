## 比赛介绍：
- 现实生活中的招牌各种各样，千变万化。针对初赛，在现实世界中，选取100类常见的招牌信息，如肯德基、麦当劳、耐克等。每类招牌挑选出10～30张图像作为训练数据，5～10张图像作为测试数据。参赛者需要根据训练集，构建算法模型，然后针对测试集进行分类，将最终的分类结果上传到比赛平台。
- 比赛链接：[百度-西交大·大数据竞赛2018——商家招牌的分类与检测](http://dianshi.baidu.com/gemstone/competitions/detail?raceId=17)

## 初赛历程
1. finetune resnet50，解冻所有层（之前试过冻结所有，以及解冻部分卷积层，出来的结果都没上90），出了一个97左右的baseline
2. 加入线上图像增强方法：1）水平平移；2）上下平移；3）随机旋转；4）随机放大缩小；5）随机倾斜
3. 加入线下图像增强：对每个张图片进行随机改变亮度，随机加高斯，以及组合随机亮度和高斯，即线下每张图片扩充到了4张，总体数据扩充到了原来的四倍
4. 在resnet的基础上加对最后一层卷积层输出feature map的attention，精确度没提升
5. 在resnet的基础上加对最后一层卷积层通道的attention，精确度仍然没有提升，很迷...不知道是不是姿势不对
6. 在resnet的基础上改用focal loss, 还是没多少提升
7. 换模型，finetune densenet169, 解冻所有层，精确度能上98
8. 在densenet169基础上, 加入focal loss,精确度提升到了99+!!!(数据是增强方式任然是前面的这些方式)
9. 增加新的数据处理方法,对所有训练集的图片进行切边, 只剩下以图片中心为中心,宽和高分别为图片原来宽高90%(这个数值可调,没时间调整了)的区域块
10. 根据分类效果不好的图片自己从网上采集图片, 新增了100张左右图片放入训练集, 结果精确度下降了0.001, 应该是采集的图片质量不好...
11. 融合模型,步骤8出来的最优的两个模型+步骤10出来的一个模型, 融合方法为对三个预测结果矩阵简单相加取值最大的位置为预测类别标签, 最后提交准确率为997.

## 文件说明
#### 一.文件结构

	.
	├── datasets
	│   └── 百度云地址
	├── finetune.py
	├── helper
	│   ├── crop_pics.py
	│   ├── datagenerator_for_trainForFlow.py
	│   ├── generate_data_for_flow.py
	│   ├── helper.py
	│   ├── train_valid_split.py
	│   └── visualize.py
	├── label_name_reference.txt
	├── model_weights
	│   └── 百度云
	├── readme
	└── truth.txt

	3 directories, 13 files

#### 二.文件说明

- datasets: 初赛图片数据,train,test; 自己收集的扩充数据的百度云地址
- finetune.py: 主函数
- helper: 辅助函数文件夹
	- crop_pics.py:切图函数
	- datagenerator_for_trainForFlow.py: 线下数据增强
	- generate_data_for_flow.py:将train中的图片数据按照标签分校文件夹
	- helper.py: 辅助函数
	- train_valid_split.py: 训练集和验证集产生函数(每类取一定比例作为验证集)
	- visualize.py: 可视化数据集
- label_name_reference.txt: 标签与店名的映射
- model_weigths: 官方预训练数据以及自己训练得到的权值文件,具体见百度云文件说明
- truth.txt: test集对应的标签,可用于线下验证模型.

#### 三.运行方式

1. 运行train_valid_split.py,得到split_train.txt和split_valid.txt
2. 运行generate_data_for_flow.py,得到train_for_flow和valid_for_flow两个文件夹(for keras .flow_from_directory)
3. 运行crop_pics.py,对train中的所有图片进行切边
4. 运行datagenerator_for_trainForFlow.py, 对train_for_flow文件夹下的所有图片进行线下增强
5. 运行主程序finetune.py进行模型的训练以及预测

## 初赛总结
- 发现比赛的时候已经是6月1号了，下单1080卡然后等卡到, 正式开始是在3号左右，初赛8号截止，所有有点仓促. 
- 经历了各种调参的折磨, 最后结果还不错
- 还有一些方法没来得及尝试: 1) attention只是简单的对最后一层卷积层加了一下,而且attention层没叠加dense; 2) 采用senet的模块,不过这个感觉就是对通道的attention,跟1应该差不多; 3)换别的模型,多跑几个模型融合泛华能力可能更好; 4)试一试OHEM, 可以更好的针对易训练错的图片; 5) 换种思路,用ocr的方式来解决这个分类问题

## 遇到的问题以及解决方法,以及学到的新姿势

- 在我的实验中,finetune的时候全解冻比部分或者全冻结的效果好
- batch_size小一点比较好
- CNN加attention的时候softmax需要指定维度，因为默认为-1维度，即通道这个维度，这样的话默认是对通道进行attention了。为了对featuremap进行attention需要将featuremap reshape一下，然后指定维度进行softmax

## 一些参考

#### Focal Loss

- 论文:[Focal Loss for Dense Object Detection
](https://arxiv.org/pdf/1708.02002.pdf)
- 论文笔记:
	- [笔记1](https://blog.csdn.net/qq_34564947/article/details/77200104)
	- [笔记2](https://blog.csdn.net/u014380165/article/details/77019084
)
- 参考代码:
	- [focal-loss-keras（二分类）](https://github.com/mkocabas/focal-loss-keras)
	- [Focal-Loss-implement-on-Tensorflow (tensorflow, Multilabel) ](https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py)
	- [FocalLoss_Keras (二分类) ](https://github.com/Atomwh/FocalLoss_Keras/blob/master/focalloss.py)

#### OHEM

- [how to do OHEM(online hard example mining) on keras](https://github.com/keras-team/keras/issues/6569)

#### senet

- 论文:[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [专栏 | Momenta详解ImageNet 2017夺冠架构SENet
](http://www.sohu.com/a/161633191_465975
- 参考代码: [SENet-Tensorflow](https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_ResNeXt.py)
)
