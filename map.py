
 
#from keras.models import Model
#from cv2 import cv2 as cv
#import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.layers import Activation
#from pylab import *
#import keras  
#from keras import backend as K  
#K.set_image_dim_ordering('th')
#import os
#os.environ['KERAS_BACKEND']='tensorflow'


#def get_row_col(num_pic): 
#    squr = num_pic ** 0.5 
#    row = round(squr) 
#    col = row + 1 if squr - row > 0 else row
#    return row, col
     
#def visualize_feature_map(img_batch): 
#    feature_map = np.squeeze(img_batch, axis=0) 
#    print(feature_map.shape) 
#    feature_map_combination = []
#    plt.figure()
#    num_pic = feature_map.shape[2] 
#    row, col = get_row_col(num_pic) 
#    for i in range(0, num_pic): 
#        feature_map_split = feature_map[:, :, i] 
#        feature_map_combination.append(feature_map_split) 
#        plt.subplot(row, col, i + 1) 
#        plt.imshow(feature_map_split) 
#        axis('off') 
#        title('feature_map_{}'.format(i)) 

#    plt.savefig('feature_map.png') 
#    plt.show() 
#    #plt.waitforbuttonpress()
#    # 各个特征图按1：1 叠加 
#    feature_map_sum = sum(ele for ele in feature_map_combination) 
#    plt.imshow(feature_map_sum) 
#    #plt.waitforbuttonpress()
#    plt.savefig("feature_map_sum.png") 
    

#def create_model(): 
#    model = Sequential()
#    # 第一层CNN 
#    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小 
#    model.add(Convolution2D(9,(5,5), activation = 'relu',input_shape=img.shape))
   
#    #model.add(Activation('relu')) 
#    model.add(MaxPooling2D(pool_size=(4, 4))) 
#    # 第二层CNN
#    model.add(Convolution2D(9,(5,5), activation = 'relu',input_shape=img.shape))
#    #model.add(Activation('relu')) 
#    model.add(MaxPooling2D(pool_size=(3, 3))) 
#    # 第三层CNN 
#    model.add(Convolution2D(9,(5,5), activation = 'relu',input_shape=img.shape)) 
#    #model.add(Activation('relu')) 
#    model.add(MaxPooling2D(pool_size=(2, 2))) 
#    # 第四层CNN 
#    model.add(Convolution2D(9,(3,3), activation = 'relu',input_shape=img.shape))
#    ##model.add(Activation('relu'))  
#    model.add(MaxPooling2D(pool_size=(2, 2))) 
#    return model
        
#if __name__ == "__main__":
#    img = cv.imread('1.jpg')
#    cv.imshow('image',img)
#    cv.waitKey(0)
#    model = create_model() 
#    img_batch = np.expand_dims(img, axis=0) 
#    conv_img = model.predict(img_batch) # conv_img 卷积结果 
#    visualize_feature_map(conv_img)

# coding: utf-8
 
from keras.models import Model  
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
 

 #输入要绘制的子图像个数，计算绘图几行几列
def get_row_col(num_pic):
    squr = num_pic ** 0.5   #开根号
    row = round(squr)       #四舍五入
    col = row + 1 if squr - row > 0 else row   #计算行列数 
    return row, col
 
#绘图函数，输入卷积结果列表
def visualize_feature_map(img_batch): 
    feature_map = np.squeeze(img_batch, axis=0)  #删除第一个维度的列
    print(feature_map.shape)
 
    feature_map_combination = [] #定义一个空的列表
    plt.figure()                 #绘图
 
    num_pic = feature_map.shape[2] #特征图的个数
    row, col = get_row_col(num_pic) #获取需要绘图需要的行列数
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]  #分离出各个特征图
        feature_map_combination.append(feature_map_split) #将特征图添加进列表
        plt.subplot(row, col, i + 1) #在图像上绘制第i个特征图
        plt.imshow(feature_map_split) #plt.imshow()函数负责对图像进行处理，并显示其格式
        axis('off') #关闭坐标轴显示
        title('feature_map_{}'.format(i)) #每一个特征图的标题
 
    plt.savefig('feature_map_test.png')  #保存绘制好的图像
    plt.show()#显示图像
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination) # list生成式：用[...for...in...]语句生成list，叠加特征图
    plt.imshow(feature_map_sum) 
    plt.savefig("feature_map_sum_test.png")
    plt.show()
 
    #创建模型
def create_model():
    model = Sequential()  #创建顺序模型
 
    # 第一层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape)) #添加卷积层，9是输出维度，5*5的滤波器，输入图像参数
    model.add(Activation('relu'))#激活函数
    model.add(MaxPooling2D(pool_size=(4, 4)))#最大池化
 
    # 第二层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
 
    # 第三层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    # 第四层CNN
    model.add(Convolution2D(9, 3, 3,  border_mode='same',input_shape=img.shape))#same图像边缘填充0便于卷积
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
 
    return model
 
 
if __name__ == "__main__":
    img = cv2.imread('5.jpg')  #opencv读取图像
    model = create_model()     #创建模型
    img_batch = np.expand_dims(img, axis=0) #在0维处添加图像数据
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    visualize_feature_map(conv_img) #绘制卷积结果图
    







##第二段
## coding: utf-8
#from keras.applications.vgg19 import VGG19
#from keras.preprocessing import image
#from keras.applications.vgg19 import preprocess_input
#from keras.models import Model
#import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
 
 
#def get_row_col(num_pic):
#    squr = num_pic ** 0.5
#    row = round(squr)
#    col = row + 1 if squr - row > 0 else row
#    return row, col
 
 
#def visualize_feature_map(img_batch):
#    feature_map = img_batch
#    print(feature_map.shape)
 
#    feature_map_combination = []
#    plt.figure()
 
#    num_pic = feature_map.shape[2]
#    row, col = get_row_col(num_pic)
 
#    for i in range(0, num_pic):
#        feature_map_split = feature_map[:, :, i]
#        feature_map_combination.append(feature_map_split)
#        plt.subplot(row, col, i + 1)
#        plt.imshow(feature_map_split)
#        axis('off')
 
#    plt.savefig('feature_map_vgg19.png')
#    plt.show()
 
#    # 各个特征图按1：1 叠加
#    feature_map_sum = sum(ele for ele in feature_map_combination)
#    plt.imshow(feature_map_sum)
#    plt.savefig("feature_map_sum_vgg19.png")
 
 
#if __name__ == "__main__":
#    base_model = VGG19(weights='imagenet', include_top=False)
#    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
#    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
#    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
#    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
 
#    img_path = '1.jpg'
#    img = image.load_img(img_path)
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#    block_pool_features = model.predict(x)
#    print("123:"+block_pool_features.shape)
 
#    feature = block_pool_features.reshape(block_pool_features.shape[1:])
 
#    visualize_feature_map(feature)
