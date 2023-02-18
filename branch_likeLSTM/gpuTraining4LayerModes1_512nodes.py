########################################################################################################################
print("0.这是当前程序的分支，用于训练带有上个时刻数据的数据样本集")
print("0.这是简化程序，原始带有更多测试和原始模型的程序在mainTestCSVMLP3(hmcnf_keras).ipynb")
print("0.这是简化程序，只训练和测试5label模型,编号为0")
print("0.这是简化程序，把神经网络转为4层模型看，看能否提高正确率")
print("0.tensorflow版本为2.3.0,python为3.6和3.8")
print("pip install pandas numpy matplotlib sklearn copy imblearn pickle ")
########################################################################################################################
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import tree
#import graphviz 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import copy
from imblearn.over_sampling import RandomOverSampler

import pickle  

########################################################################################################################
#######开始为功能函数
#######函数用于决策树分析

def dtFitAndSave(x,y,saveName):
    str1="dtFitAndSave,用于决策树拟合和识别"
    
    dt = tree.DecisionTreeClassifier(max_depth=7,min_samples_leaf=100)
    dt = dt.fit(x, y)
    tree.plot_tree(dt)
    #data=tree.export_graphviz(dt, out_file=None,class_names=None,filled=True) 
    #graph = graphviz.Source(data)
    #graph.render(saveName)
    
    yPredict = dt.predict(x)
    tmp1 = classification_report(y,yPredict)
    print("纯决策树的识别\n",tmp1)
    mat1num = confusion_matrix(y,yPredict)
    mat2acc = confusion_matrix(y,yPredict,normalize='pred')
    print(mat1num)
    print(np.around(mat2acc , decimals=3))
    #text_representation = tree.export_text(dt)
    #print(text_representation)
    #yPredict = dt.predict_proba(x)
    #index = np.where((yPredict[:,1]<0.98)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    #index = np.where((yPredict[:,1]<0.90)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    #index = np.where((yPredict[:,1]<0.80)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    #index = np.where((yPredict[:,1]<0.70)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    return dt,yPredict

########################################################################################################################
###简单模型3，resnet_like

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
    str1="kerasFitAndSaveSimple4LayerLikeResnet,用于resnet_like的神经网络拟合和识别"
    
    nSamples,features_size = x.shape
    relu_size = 512
    dropout_rate = 0.05
    hierarchy = [1,1,1,1]#4层，相对与默认3层，对于当前数据集是否能提高正确率
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
    
    try:
       build_model = keras.models.load_model(saveName)
    except:
       print("Error:keras.models.load_model(%s):" %saveName)
       input()
    #build_model.fit([x],[yOneHot],epochs=10, batch_size=10000*1)
    
    #checkpoint_filepath = './checkpoints'
    
    my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./logs/model.{epoch:05d}.h5',monitor='accuracy'),
    #tf.keras.callbacks.ModelCheckpoint(filepath='./logs/model.{epoch:05d}-{accuracy:.4f}.h5',monitor='accuracy'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    build_model.fit(x,yOneHot,epochs=150, batch_size=40000*1,callbacks=my_callbacks)#GPU用这个
    #saveName = "KerasSimple3_likeResnet.h5"
    build_model.save(saveName)
    #plot_model(build_model, to_file='KerasSimple3_likeResnet.png', show_shapes=True)
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

def test1():
    ########################################################################################################################
    ########################################################################################################################
    print("0.这是简化程序，把神经网络转为4层模型看，看能否提高正确率")
    print("0.这是当前程序的分支，用于训练带有上个时刻数据的数据样本集")
    ########################################################################################################################

    print("reading data france,读取数据并且把数据进行onehot处理")
    #file1 = "../trainData/france_0_allSamples1.csv"
    file1 = "../trainData/france_0_allSamples1_2slot.csv"#新数据集合，样本带有上一个时刻数据

    xyDataTmp = pd.read_csv(file1)
    print(xyDataTmp.info())
    xyData = np.array(xyDataTmp)
    h,w = xyData.shape
    #print("xyData[0,:]",xyData[0,:])
    #nput()

    #这里比较怪，因为我直接把上一个时刻数据的40个特征，加到原始样本的最后面
    labelPos = 49
    y0rigin  = xyData[:,labelPos].copy()#第49列为标志位
    xyData[:,labelPos] = 0#标志位置0
    #补丁
    x0rigin = xyData[:,1:w-1]#用所有的数据,除去首列的车ID
   
    x0rigin[:,6] = [string2int(inputString) for inputString in x0rigin[:,7] ]#字符串vehLaneID 变为整数
    #print(x0rigin[0,:])
    #input()
    
   
    x0rigin =x0rigin.astype(np.float32)#GPU 加这个
    y0rigin =y0rigin.astype(np.int64)#GPU 加这个
    #rint(x0rigin[0:10,:])
    #rint(y0rigin[0:10])
    #nput()
    ros = RandomOverSampler(random_state=0)
    x0,y0= ros.fit_resample(x0rigin , y0rigin)#对数据不平衡进行处理，保证样本数一致

    x0=x0.astype(np.float32)#GPU 加这个
    y0=y0.astype(np.int64)#GPU 加这个
    yl5 = y0
    print("x0.shape:",x0.shape,"y0.shape:",y0.shape,"y0.type:", type(y0) )
    del xyDataTmp #节省内存
    del xyData #节省内存




    ########################################################################################################################
    print("准备字典，用于保存训练后的数据")

    xFloors=  dict()
    yFloors =  dict()
    dtModeFloors=  dict()
    dtPredictLabel = dict()
    kerasPredictLabel = dict()
    kerasModelNameFloors =dict()
    encFloors= dict()
    ########################################################################################################################
    ###现在暂时不训练多层模型，只训练5label模型
    if 1:
        print("5label 模型")
        x=x0
        y=yl5


        x=x.astype(np.float32)#GPU 加这个
        y=y.astype(np.int64)#GPU 加这个
        print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )

        num_labels = 5 
        nSamples,nFeatures =  x.shape
        enc = OneHotEncoder()
        y= y.reshape(nSamples,-1)

        print("y.shape:",y .shape,"y.type:", type(y) )
        enc.fit(y)
        yOneHot=enc.transform(y).toarray()
        
        saveName = "./model.h5"
                    
        if 1:
            kerasModel3_5label = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
        yKeras_5label=getKerasResnetRVL(x,enc,saveName)

        print('keras\n')
        mat1num = confusion_matrix(y, yKeras_5label)
        mat2acc = confusion_matrix(y, yKeras_5label,normalize='pred')
        print('mat1num\n',mat1num)
        print('mat2acc\n',np.around(mat2acc , decimals=3))




        dt_5label,dt_PredictLabel = dtFitAndSave(x,yl5,"5label")
        enc_5label = enc

        xFloors[0] =  x.copy()
        yFloors[0] =  y.copy()
        dtModeFloors[0] =  dt_5label
        dtPredictLabel[0] = dt_PredictLabel.copy()
        kerasPredictLabel[0] = yKeras_5label.copy()
        kerasModelNameFloors[0] =saveName
        encFloors[0] = enc_5label

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################



def main():
    print("测试程序,运行20次,20x12min")
    test1()
if __name__ == "__main__":
    main()
