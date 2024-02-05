
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
otherVeh = name2+name3+name4+name5+name6_error 
headName2SlotX94 = headName48+name1C+name2+name3+name4+name5+name6_error #48+40+7+39 =  94
print("2slot的数据列表为：headName2SlotXY96\n")
print("去掉vehicleID,2slot的X输入数据列表为：headName2SlotX94\n")
print("第二层模型的x的输入为103或者108：headName2SlotX94+SUMO动态特征")
print("sumoSimDataLevel7.csv里面的sampleIndex，相对于步骤1的lowprobSamplesLevel%d.pkf")
print("stage2ForMainSimpleStep3.pkf里面的xOriginSumoAdded（也就是X），用于训练，不对应sumoSimDataLevel7.csv里面的sampleIndex")





############################################################################################################
############################################################################################################
############################################################################################################
print("程序4: 在论文中体现，实现验证幽灵堵车规则")
print("接程序3:程序3为根据输入的SUMO模拟特征对低概率样本进行重新训练,从而获得识别正确率提升")
print("mainSimpleStep4为简化的步骤4程序，注意只能接mainSimpleStep3的输出结果")

print(" 预测幽灵堵车.原因为本来这条道路机容易出现幽灵堵车，和有车不停变道。主要为D规则 \
\nA.如果蒙特卡洛预测能高速通过路口，但是预测minSpeed比较低，同时如果设定车辆延迟比较高时，也不能通过路口，认为出现幽灵堵车 \
\nB.如果初始模型能预测通过路口，但是加入道路特征和设定车辆延迟比较高时也不能通过路口，也不能通过路口，认为出现幽灵堵车 \
\nC.如果初始模型能预测能通过路口，但是概率比较低，但是加入道路特征和设定车辆延迟比较高时不能通过路口，认为出现幽灵堵车 \
\nD.如果初始模型和蒙特卡洛模拟预测都能通过路口，但是概率比较低，但是加入道路特征和设定车辆延迟比较高时不能通过路口，认为出现幽灵堵车 \
\nE.如果初始模型和蒙特卡洛模拟预测都能通过路口，但是概率比较高，但是加入道路特征和设定车辆延迟比较高时不能通过路口，不知道。\
\nF.如果原始模型和SUMO增强模型都预测能高速通过路口，但是概率比较低而SUMO模拟预测为不能通过路口，那是什么呢？？也许是提前减速，也就是减低最大速度。")

########################################################################################################
'''将输出重定向到文件'''
import sys 
fs1 = open('printlog.txt', 'w+')
sys.stdout = fs1  # 将输出重定向到文件

########################################################################################################
level = 7

if level==7:
    dfSumoData = pd.read_csv('./data/sumoSimDataLevel7.csv', sep=',')

    headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                                   'smv1','smv2']
    xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                            'outputAvgSpeed','smv1','smv2']
    
    dfSimVehParams = pd.read_csv('./data/paramsVehAllLevel7.csv', sep=',')
    
if level==2:
    
    dfSumoData = pd.read_csv('./data/sumoSimDataLevel2.csv', sep=',')

    headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3',\
                                                   'smv1','smv2']
    
    xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','outputAvgSpeed','smv1','smv2']
    
    dfSimVehParams = pd.read_csv('./data/paramsVehAllLevel2.csv', sep=',')





headSimVehParams = ['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                       'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                               'minSpeed0']


#fpk=open('stage2-LowprobSamples-addingSumoFeatures-Hierachical.pkf','rb')   
#[xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors]=pickle.load(fpk)  
#fpk.close()  

fpk=open('stage2ForMainSimpleStep3.pkf','rb') 
[xOriginSumoAdded,yOriginSumoAdded,saveName,enc,x_train,y_train,yKerasSumoPredict]=pickle.load(fpk)  

#np.concatenate([lowproKerasStage1Input,x1_sumoOutput,x2_yKerasOutput,x3_outputListNN,x4_outputAvgSpeed,x5_sumoOutList],axis=1)#71%
print('xOriginSumoAdded.shape:',xOriginSumoAdded.shape)
xInputDF = pd.DataFrame(xOriginSumoAdded,columns=xInputHeader)

fs = "./data/step4_xInput_level%d.csv" %level
xInputDF.to_csv(fs,index= False)

model_name = saveName 
model = keras.models.load_model(model_name)
yKerasSumo= model.predict([xOriginSumoAdded], batch_size=2560)
        
nSamples,nFeatures = xOriginSumoAdded.shape

print("xOriginSumoAdded.shape:",xOriginSumoAdded.shape)
print("dfSumoData.shape:",dfSumoData.shape)
print("dfSimVehParams.shape:",dfSimVehParams.shape)   


ghostNum1  = 0  
ghostNum2  = 0 
flag0Num1  = 0
simpleFeatures1 = np.array([])
simpleFeatures2 = np.array([])
for j in range(nSamples):
    print('#################################################################')  
    print('j:',j) 
    '''xOriginSumoAdded（也就是X），用于训练，行标号不对应sumoSimDataLevel7.csv里面的sampleIndex'''
    '''根据xOriginSumoAdded行标号，获得sumoSimDataLevel7.csv行标号对应的sampleIndex'''
    yNN =  yKerasSumo[j]
    #print('yNN:',np.round(yNN,2))
    yKerasSumoFlag = np.argmax(yNN)
    if yKerasSumoFlag>0:
        continue
    
    dataSD = dfSumoData.iloc[j]
    sampleIndex = dataSD['sampleIndex'].item()
    dataVP = dfSimVehParams[dfSimVehParams['sampleIndex'] == sampleIndex]
    yNN =  yKerasSumo[j]
    print('yNN:',np.round(yNN,2))


    #进行分析
    xtmp = xOriginSumoAdded[j]
    vehs = xtmp[8:48]
    vehs = vehs.reshape(-1,2)
    vehs = vehs[np.where(vehs[:,0]>0)]
    vehsOthers1 = vehs[0:-1]#最后一个是目标车，不要
    numFrontVeh,tmp = vehsOthers1.shape
    redTimeLeft = xtmp[0]
    dist = xtmp[1] 
    speed = xtmp[2]
    maxSpeed = xtmp[3]

   
    kerasPredictLabel  = dataSD['kerasPredictLabel'].item()
    sumoOutputSpeedTag  = dataSD['sumoOutputSpeedTag'].item()
   
    print(dataSD[['originOutput','kerasPredictLabel','sumoOutputSpeedTag']])#item对于Series
    #tmp = x_train[j][0:48]
    #print("x:",np.round(tmp,2))
    print("yKerasSumoFlag:",yKerasSumoFlag)
    
    #如果初始模型和蒙特卡洛模拟预测都能通过路口，但是概率比较低，
    #但是加入道路特征和设定车辆延迟比较高时不能通过路口，认为出现幽灵堵  
    yKerasSumoFlag = np.argmax(yNN)
    print("yKerasSumoFlag:",yKerasSumoFlag)
    
    ###论文中解释幽灵堵车情况
    if yKerasSumoFlag == 0: 
        flag0Num1 = flag0Num1+1
        
        #认为遇到了幽灵堵车现象，分析进行
        dataVPTmp1=dataVP
        dataVPTmp1 = dataVPTmp1[['minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed']]
        dataVPTmp1 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 40/3.6) & (dataVPTmp1['maxSpeed'] > 40/3.6)]
        dataVPTmp1 = dataVPTmp1[(dataVPTmp1['reacTime0'] <1.2) & (dataVPTmp1['reacTime'] <1.2)]
      
        meanSpeed = dataVPTmp1['minSpeed0'].mean()
        if  numFrontVeh >0 and sumoOutputSpeedTag  > 0:
            ghostNum1 = ghostNum1+1
        
        if  numFrontVeh >0 and  meanSpeed>= 5/3.6:
            ghostNum2 = ghostNum2+1
            
     
        
        if  numFrontVeh >0 and meanSpeed  >= 5/3.6:
            f1 = numFrontVeh*2+redTimeLeft
            f2 = dist/speed
            simpleFeatures2 = np.append(simpleFeatures2,[f1,f2,f1/f2])
        
        if numFrontVeh >0 and meanSpeed  < 5/3.6:
            f1 = numFrontVeh*2+redTimeLeft
            f2 = dist/speed
            simpleFeatures1 = np.append(simpleFeatures1,[f1,f2,f1/f2])
           
        
print('#################################################################') 
strTmp1 = "flag0Num1:%d,ghostNum1:%d,value:%3f" %(flag0Num1,ghostNum1,ghostNum1*1.0/flag0Num1)
strTmp2 = "flag0Num1:%d,ghostNum2:%d,value:%3f" %(flag0Num1,ghostNum2,ghostNum2*1.0/flag0Num1)
print(strTmp1)
print(strTmp2)


            
            
