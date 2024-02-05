########################################################################################################################
print("接程序2: 综合SUMO输出，对keras输出进行优化。程序输入为程序2的输出")
print("优化选择1.NN。 2 回归分析。3 概率分析。")
print("程序编号为3(临时)")
########################################################################################################################
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
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes

from sklearn.metrics import accuracy_score



########################################################################################################################
###简单模型3，resnet_like
def local_model(num_labels, dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_labels, activation='sigmoid'))
    return model

def global_model(dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    return model



def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid',name="global"))
    return model

def softmax_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='softmax',name="global"))
    return model

def getKerasResnetRVL(x,enc,saveName):
    print(saveName)
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



########################################################################################################################
##手工确定层次结构，以前测试时候为5层，根据论文为9层
def convertY2Hieral(y):
    #mat2acc
    # [[0.914 0.009 0.017 0.007 0.032 0.    0.    0.    0.   ]
    # [0.027 0.984 0.006 0.007 0.018 0.    0.    0.    0.   ]
    # [0.02  0.006 0.972 0.    0.011 0.    0.    0.    0.   ]
    # [0.036 0.002 0.    0.986 0.014 0.    0.002 0.    0.   ]
    # [0.003 0.    0.    0.    0.925 0.    0.    0.    0.   ]
    # [0.    0.    0.    0.    0.    1.    0.005 0.    0.   ]
    # [0.    0.    0.    0.    0.    0.    0.993 0.    0.004]
    # [0.    0.    0.    0.    0.    0.    0.    0.996 0.   ]
    # [0.    0.    0.004 0.    0.    0.    0.    0.004 0.996]]
    
    
    #hierarchy = [2,4,6,8,9]
   # labelDict = {"0":["01234","0123","012","01","0"],\
   #               "1":["01234","0123","012","01","1"],\
   #               "2":["01234","0123","012","2","2"],\
   #               "3":["01234","0123","3","3","3"],\
   #              "4":["01234","4",    "4","4","4"],\
   #              "5":["5678","5",     "5","5","5"],\
   #              "6":["5678","678",   "67","6","6"],\
   #              "7":["5678","678",   "67","7","7"],\
   #              "8":["5678","678",   "8","8","8"],\
   #               }
    
    hierarchy = [2,3,4,5,6,7,8,9]
    labelDict = {"0":["01234",        "01234",        "01234",   "01234",      "0123","012","01","0"],\
                  "1":["01234",        "01234",        "01234",  "01234",     "0123","012","01","1"],\
                  "2":["01234",          "01234",      "01234",  "01234",     "0123","012","2","2"],\
                  "3":["01234",         "01234",       "01234",  "01234",    "0123","3","3","3"],\
                 "4":["01234",          "01234",       "01234",  "01234" ,     "4", "4","4","4"],\
                 "5":["5678",               "5",        "5" ,      "5",       "5", "5","5","5"],\
                 "6":["5678",            "678",        "6",        "6",       "6", "6","6","6"],\
                 "7":["5678",             "678",       "78",       "7",       "7", "7","7","7"],\
                 "8":["5678",             "678",       "78" ,      "8",        "8", "8","8","8"],\
                  }
    '''
    hierarchy = [5,9]
    labelDict = {"0":["01","0"],\
                  "1":["01","1"],\
                  "2":["2","2"],\
                  "3":["34","3"],\
                "4":["34","4"],\
                 "5":["56","5"],\
                 "6":["56","6"],\
                 "7":["78","7"],\
                 "8":["78","8"],\
                 }
    '''

    y1 = [list(labelDict[str(x)]) for x in y]
   
    #print("!!!y1.type:", type(y1))
    #print(y1[:2])
    #y2 = [t1[0] for t1 in y1]
    #print(len(y2))
  

    return y1,hierarchy 

    # 当前车道，每个红灯车的所有时刻的样本
name1A = ["vehID", "redLightTime", "distToRedLight", "speed", "laneAvgSpeed",
         "arriveTime1", "arriveTime2", "vehLaneID", "ArrTimeDivRedTime"]#9
name1B = ["redLightTime", "distToRedLight", "speed", "laneAvgSpeed",
         "arriveTime1", "arriveTime2", "vehLaneID", "ArrTimeDivRedTime"]#8
name1C = ["redLightTime", "distToRedLight", "speed", "laneAvgSpeed",
         "arriveTime1", "arriveTime2","ArrTimeDivRedTime"]#7，去掉"vehID"，"vehLaneID"
name2 = ["vehPos_1", "vehSpeed_1", "vehPos_2", "vehSpeed_2",
         "vehPos_3", "vehSpeed_3", "vehPos_4", "vehSpeed_4"]
name3 = ["vehPos_5", "vehSpeed_5", "vehPos_6", "vehSpeed_6",
         "vehPos_7", "vehSpeed_7", "vehPos_8", "vehSpeed_8"]
name4 = ["vehPos_9", "vehSpeed_9", "vehPos_10", "vehSpeed_10",
         "vehPos_11", "vehSpeed_11", "vehPos_12", "vehSpeed_12"]
name5 = ["vehPos_13", "vehSpeed_13", "vehPos_14", "vehSpeed_14",
         "vehPos_15", "vehSpeed_15", "vehPos_16", "vehSpeed_16"]
name6 = ["vehPos_17", "vehSpeed_17", "vehPos_18", "vehSpeed_18",
         "vehPos_19", "vehSpeed_19", "vehPos_20", "vehSpeed_20"]

name6_error = ["vehPos_17", "vehSpeed_17", "vehPos_18", "vehSpeed_18",
         "vehPos_19", "vehSpeed_19", "vehPos_20"] #原始数据出现错误，
vehAll = name2+name3+name4+name5+name6 #40
headName49 = name1A+vehAll
headName48 = name1B+vehAll

headName2SlotX95 = headName49+name1C+name2+name3+name4+name5+name6_error #9+40+7+39= 95
headName2SlotXY96 = headName2SlotX95+['minSpeedFlag'] ##96

headName2SlotX94 = headName48+name1C+name2+name3+name4+name5+name6_error #48+40+7+39 =  94
print("2slot的数据列表为：headName2SlotXY96")
print("去掉vehicleID,2slot的X输入数据列表为：headName2SlotX94")
print("第二层模型的x的输入为103或者108：headName2SlotX94+SUMO动态特征")
print("sumoSimDataLevel7.csv里面的sampleIndex，相对于步骤1的lowprobSamplesLevel%d.pkf")
print("stage2ForMainSimpleStep3.pkf里面的xOriginSumoAdded，不对应sumoSimDataLevel7.csv里面的sampleIndex")
############################################################################
####HMCM-F ,层次模型，发现hmcn-f训练效果很差，所以采用分离式
###每一层的识别模型都是4层模型
##分层重新训练，加入特征SMV1,SMV2
def sepHier1_SUMO(x,yOneHot,num_labels,saveName,levelIndex,numLayers,numEpochs = 10,srelu_size = 256,dropout_rate = 0.05):
    
    str1="layIndex-"+str(levelIndex)
    
    nSamples,features_size = x.shape
    relu_size = 256
    dropout_rate = 0.05
    global_models = []
    
    label_size = num_labels
    featuresInput = layers.Input(shape=(features_size,))
    features = layers.BatchNormalization()(featuresInput)
    #features=featuresInput
    for i in range(numLayers):
        if i == 0:
            global_models.append(global_model(dropout_rate, relu_size)(features))
        else:
            global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))
    
    p_glob = softmax_model(label_size)(global_models[-1])
    build_model = tf.keras.Model(inputs=[featuresInput], outputs=[p_glob])

    
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    if 0:
        build_model = keras.models.load_model(saveName)
    if 1:#用于画图
        #build_model.fit([x],[yOneHot],epochs=1, batch_size=10000*1)
        #build_model.summary()
        build_model.fit(x,yOneHot,epochs=1, batch_size=10000*1)
        plot_model(build_model, to_file=str1+".jpg", show_shapes=True)
    
  
    build_model.fit(x,yOneHot,epochs=numEpochs,batch_size=160000*1)#GPU用这个
    build_model.save(saveName)
    return build_model



########################################################################################################################   
########################################################################################################################    
print("##############################################################################################################")

print("程序编号为3.1，主程序开始运行")

####原始的keras训练数据中的低概率数据，阶段1
#sample_name = 1(ID)+8(keyFeature)+40(otherVehcle)+6(keyFeatures)+40(otherVehs)+1(flag)= 96
#xlowpra:x-name = 8(keyFeature)+40(otherVehcle)+6(keyFeatures)+40(otherVehs)= 94
level = 7

strTmp = 'lowprobSamplesLevel%d.pkf' %level
fpk=open(strTmp,'rb')   
[xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
print("xlowpra",xlowpra.shape)
fpk.close()      



####原始的keras训练数据，阶段2
strTmp = './data/sumoSimDataLevel%d.csv' %level
df = pd.read_csv(strTmp, sep=',')

print("sumoSimData.csv",df.shape)
print("sumoSimData.csv",df.columns)
numSamples,numFeatures = df.shape


    

sumoOutput='sumoOutputSpeedTag'
yKerasOutput='kerasPredictLabel'
originOutput ='originOutput'
sumoOutList = ['smv1','smv2']
outputListNNall = ['NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8']
numNN = ylowpraPredictNN.shape[1]
outputListNN = outputListNNall[0:numNN]
outputAvgSpeed = 'outputAvgSpeed'

'''根据成功仿真的样本标号，这里对原始lowprobIndexd进行了筛选,用于后期加入原始特征'''
df1 = df[ "sampleIndex"]
lowprobIndex = df1.iloc[0:numSamples].to_numpy()
xlowpra2 = xlowpra[lowprobIndex]#这里对原始lowprobIndexd进行了筛选

print('xlowpra2.shape:',xlowpra2.shape)
#print('xlowpra2[0]:',xlowpra2[0])

##############################################################################
df1 = df[sumoOutput]
x1_sumoOutput = df1.iloc[0:numSamples].to_numpy().reshape(-1,1)


df1 = df[yKerasOutput]
x2_yKerasOutput = df1.iloc[0:numSamples].to_numpy().reshape(-1,1)

df1 = df[outputListNN]
x3_outputListNN = df1.iloc[0:numSamples].to_numpy()

df1 = df[outputAvgSpeed]
x4_outputAvgSpeed = df1.iloc[0:numSamples].to_numpy().reshape(-1,1)


df1 = df[sumoOutList]
x5_sumoOutList = df1.iloc[0:numSamples].to_numpy()

df1 = df[originOutput]
yOriginOutput = df1.iloc[0:numSamples].to_numpy().reshape(-1,1)








################################################################################################################################
################################################################################################################################ 

print("#############################\n原生keras\n")
if 1:
    yPredict = x2_yKerasOutput
    tmp1 = classification_report(yOriginOutput,yPredict)
    mat1num = confusion_matrix(yOriginOutput,yPredict)
    mat2acc = confusion_matrix(yOriginOutput,yPredict,normalize='pred')
    print(tmp1)
    print(mat1num)
    print(np.around(mat2acc , decimals=3))


    score = accuracy_score(yPredict, yOriginOutput)
    print(score) 
    
    df = pd.DataFrame(np.around(mat2acc , decimals=3))
    fs = "./data/低概率样本的原始Level%dkerasNN模型的混淆矩阵结果.csv" %level
    df.to_csv(fs,index= False, header= False)


################################################################################################################################
################################################################################################################################   
print("\n#############################加入新特征SUMO+阶段1的特征,对低概率样本重新训练多级独立kerasNN")

#基于lowprobIndex对原始低概率样本进行了筛选
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''开始训练第二层模型前的数据预处理'''

'''将场景静态特征和SUMO动态特征进行的整合'''
x = np.concatenate([xlowpra2,x1_sumoOutput,x2_yKerasOutput,x3_outputListNN,x4_outputAvgSpeed,x5_sumoOutList],axis=1)#71%

 
print('x.shape:',x.shape)

rN,cN= np.where(np.isnan(x))
for i in range(rN.shape[0]):
    x[rN[i],cN[i]] = 0
     

y=yOriginOutput.reshape(1,-1)[0]
x=x.astype(np.float32)#GPU 加这个
y=y.astype(np.int64)#GPU 加这个
print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )


#保存原始全部数据
xOriginSumoAdded   = x
yOriginSumoAdded  = y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=0)

nSamples,nFeatures =  x_train.shape
 
tmp = x_train[0][0:48]
print("x_train[0]:",np.round(tmp,2))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''开始训练第二层模型'''


numEpochs =2000          #1500/60/60*5 = 2houer
numLayers = 4
enc = OneHotEncoder()
nSamples,nFeatures =  x_train.shape
       
        
y_train= y_train.reshape(nSamples,-1)
enc.fit(y_train)
        
yOneHot=enc.transform(y_train).toarray()
print("encInfo:",enc.categories_,enc.get_feature_names())
print("enc.get_feature_names().shape,numLabel:",enc.get_feature_names().shape[0])
print("yOneHot[:1]:",yOneHot[:1])
        
        
#num_labels = hierarchy[i] #低概率样本下错误
num_labels = enc.get_feature_names().shape[0] #低概率样本下错误
print("num_labels:", num_labels)

numLayers = 4
levelIndex = 7
saveName = "modelSepStage2ForMainSimpleStep3.h5"
print("saveName:",saveName)

print("#############################\n加入新特征SUMO+阶段1的特征,对低概率样本重新训练多级独立kerasNN的进行分析（开始训练完成）\n")

build_model = sepHier1_SUMO(x_train,yOneHot,num_labels,saveName,levelIndex,numLayers,numEpochs)

print("#############################\n加入新特征SUMO+阶段1的特征,对低概率样本重新训练多级独立kerasNN的进行分析（已经训练完成）")

######################
print("只做最低层，对加入SUMO特征的全体低概率样本进行识别的结果")


yPredict=getKerasResnetRVL(x,enc,saveName)#输出格式为['0','1']
yKerasSumoPredict = yPredict


print("yPredict.shape:",yPredict.shape)
print("yPredict[0:10]:",yPredict[0:10])

tmp1 = classification_report(yOriginOutput,yPredict)
print(tmp1)
mat1num = confusion_matrix(yOriginOutput,yPredict)
print(mat1num)
mat2acc = confusion_matrix(yOriginOutput,yPredict,normalize='pred')  
print(np.around(mat2acc , decimals=3))

score = accuracy_score(yOriginOutput,yPredict)
print(score)   
    
    
fpk=open('stage2ForMainSimpleStep3.pkf','wb')  
pickle.dump([xOriginSumoAdded,yOriginSumoAdded,saveName,enc,x_train,y_train,yKerasSumoPredict],fpk)  
fpk.close() 



