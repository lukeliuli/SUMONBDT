#coding=gbk
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import  plot_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import tree

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import copy
from imblearn.over_sampling import RandomOverSampler

import pickle  



def global_model(dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    return model

def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid',name="global"))
    return model

def kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName):
    str1="kerasFitAndSaveSimple3LikeResnet,用于resnet_like的神经网络拟合和识别"
    
    nSamples,features_size = x.shape
    relu_size = 512
    dropout_rate = 0.05
    hierarchy = [1,1,1,1]#四层，对于当前数据集已经足够了
    global_models = []
    label_size = num_labels
    features = layers.Input(shape=(features_size,))
    for i in range(len(hierarchy)):
        if i == 0:
            global_models.append(global_model(dropout_rate, relu_size)(features))
        else:
            global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))

    p_glob = sigmoid_model(label_size)(global_models[-1])
    build_model = tf.keras.Model(inputs=[features], outputs=[p_glob])
    #model = tf.keras.Model(inputs=[features], outputs=[build_model])
    #enc = OneHotEncoder()
    #enc.fit(y)  
    #yOnehot=enc.transform(y).toarray()
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    if 0:
       build_model = keras.models.load_model(saveName)
    if 1:#用于画图
        #build_model.fit([x],[yOneHot],epochs=1, batch_size=10000*1)
        build_model.fit(x,yOneHot,epochs=1, batch_size=10000*1)
        plot_model(build_model, to_file='KerasSimple3_likeResnet_4lay512nodes.jpg', show_shapes=True)
    
  
    build_model.fit(x,yOneHot,epochs=15000, batch_size=20000*1)#GPU用这个
    #saveName = "KerasSimple3_likeResnet.h5"
    build_model.save(saveName)
    plot_model(build_model, to_file='KerasSimple3_likeResnet_4lay512nodes.jpg', show_shapes=True)
    return build_model
########################################################################################################################
def getKerasResnetRVL(x,enc,saveName):
    model_name = saveName 
    model = keras.models.load_model(model_name)
    y= model.predict([x], batch_size=2560)
    nSamples = y.shape[0]
    ###需要将预测出的值，转换01整数,并转为数字式
    for i in range(y.shape[0]):
        tmp = y[i]
        index=  np.argmax(tmp)
        y[i] = [0]*y.shape[1]
        y[i,index]=1
   

    ###  
    y= enc.inverse_transform(y)
    y= y.reshape(-1,nSamples)[0]
    
    
    return y

def string2int(inputString):
     #print(inputString)
     tmp = 0
     try:
         strTmp=[str(ord(x)) for x in inputString]
         tmp=tmp.join(strTmp)
         tmp = float(tmp)/(len(inputString)*128)
     except:
         #print(inputString)
         strTmp = inputString
         tmp= "0"
         tmp = 0
     return tmp
########################################################################################################################
########################################################################################################################
print("0.主程序开始, 建立多层嵌套决策树模型,3080ti的GPU是AMD2400CPU 运算速度100倍")
########################################################################################################################
print("读取France数据并且把数据进行onehot处理")

file1 = "../trainData/france_0_allSamples1.csv"
xyDataTmp = pd.read_csv(file1)
#print(xyDataTmp.info())
xyData = np.array(xyDataTmp)
h,w = xyData.shape
#x = xyData[:,1:23]#简单处理与SUMO数据库一致
x0rigin = xyData[:,1:w-1]#用所有的数据
y0rigin  = xyData[:,w-1]

x0rigin[:,6] = [string2int(inputString) for inputString in x0rigin[:,6] ]#字符串vehLaneID 变为整数

x0rigin =x0rigin.astype(np.float32)#GPU 加这个
y0rigin =y0rigin.astype(np.int64)#GPU 加这个


ros = RandomOverSampler(random_state=0)
x0,y0= ros.fit_resample(x0rigin , y0rigin)#对数据不平衡进行处理，保证样本数一致

x0=x0.astype(np.float32)#GPU 加这个
y0=y0.astype(np.int64)#GPU 加这个
yl5 = y0
print("x0.shape:",x0.shape,"y0.shape:",y0.shape,"y0.type:", type(y0) )
del xyDataTmp #节省内存
del xyData #节省内存




#########################################################################################################################
print("准备字典，用于保存训练后的数据")

xFloors=  dict()
yFloors =  dict()
dtModeFloors=  dict()
dtPredictLabel = dict()
kerasPredictLabel = dict()
kerasModelNameFloors =dict()
encFloors= dict()
########################################################################################################################
###现在暂时不训练多层模型，只训练9label模型
if 1:
    print("训练4层, 9 label 模型")
    x=x0
    y=yl5
    x=x.astype(np.float32)#GPU 加这个
    y=y.astype(np.int64)#GPU 加这个
    print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
    
    num_labels = 9 
    nSamples,nFeatures =  x.shape
    enc = OneHotEncoder()
    y= y.reshape(nSamples,-1)
    
    print("y.shape:",y .shape,"y.type:", type(y) )
    enc.fit(y)
    yOneHot=enc.transform(y).toarray()
    saveName = "../trainedModes/model-9label-4lays-512nodes-cpu1.h5"
    if 1:
        kerasModel3_5label = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
    yKeras_5label=getKerasResnetRVL(x,enc,saveName)
    
    print('keras\n')
    mat1num = confusion_matrix(y, yKeras_5label)
    mat2acc = confusion_matrix(y, yKeras_5label,normalize='pred')
    print('mat1num\n',mat1num)
    print('mat2acc\n',np.around(mat2acc , decimals=3))
    

