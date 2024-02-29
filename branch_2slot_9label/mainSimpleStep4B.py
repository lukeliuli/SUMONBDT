
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

import matplotlib.pyplot as plt

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
print("程序4B: 在论文中体现，实现验证不停车过路口规则,只采用一般参数下模拟的结构")
print("接程序3:程序3为根据输入的SUMO模拟特征对低概率样本进行重新训练,从而获得识别正确率提升")
print("mainSimpleStep4B注意只能接mainSimpleStep3的输出结果")


########################################################################################################
'''将输出重定向到文件'''
import sys 
fs1 = open('printlog.txt', 'w+')
sys.stdout = fs1  # 将输出重定向到文件

########################################################################################################


level = 7

if level==7:
    dfSumoData = pd.read_csv('./data/sumoSimDataLevel7_simNum500.csv', sep=',')

    headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                                   'smv1','smv2']
    xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                            'outputAvgSpeed','smv1','smv2']
    
    dfSimVehParams = pd.read_csv('./data/paramsVehAllLevel7_simNum500.csv', sep=',')
    
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


 
    
###################################################################################################\
##########################################################################################################  
'''A.获得识别和模拟合成在一起的样本的基本数据库,step4b_analy1_SSVP_level7.csv'''

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

       
nSamples,nFeatures = xOriginSumoAdded.shape

print("xOriginSumoAdded.shape:",xOriginSumoAdded.shape)
print("dfSumoData.shape:",dfSumoData.shape)
print("dfSimVehParams.shape:",dfSimVehParams.shape)   


if 1:

    analy1 =[]
    nSamples = 10000
    for j in range(nSamples):
        print('#################################################################')  
        print('j:',j) 
        '''xOriginSumoAdded（也就是X），用于训练，行标号不对应sumoSimDataLevel7.csv里面的sampleIndex'''
        '''根据xOriginSumoAdded行标号，获得sumoSimDataLevel7.csv行标号对应的sampleIndex'''



        dataSD = dfSumoData.iloc[j]
        sampleIndex = dataSD['sampleIndex'].item()
        dataVP = dfSimVehParams[dfSimVehParams['sampleIndex'] == sampleIndex]



        #进行分析
        #xOriginSumoAddedcolums = xInputHeader = 
        #
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
        arrivelTime1 = xtmp[4]
        arrivelTime2= xtmp[5]


        kerasPredictLabel  = dataSD['kerasPredictLabel'].item()
        sumoOutputSpeedTag  = dataSD['sumoOutputSpeedTag'].item()
        originOutput = dataSD['originOutput'].item()


        print(dataSD[['originOutput','kerasPredictLabel','sumoOutputSpeedTag']])#item对于Series




        #分析进行
        dataVPTmp1=dataVP
        dataVPTmp1 = dataVPTmp1[['minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed']]
        #dataVPTmp1 = dataVPTmp1[(dataVPTmp1['reacTime0'] <1.2) & (dataVPTmp1['reacTime'] <1.2)]

        dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=originOutput*5/3.6) & (dataVPTmp1['minSpeed0']<(originOutput+1)*5/3.6)] 

        index1 = j
        index2 = sampleIndex
        feature1 = [index1,index2,originOutput,dist,speed,redTimeLeft,numFrontVeh]
        featrue2 = dataVPTmp1.values.tolist()
        for i in range(dataVPTmp1.shape[0]):
            #print(featrue2[i])
            feature1 = [index1,index2,originOutput,dist,speed,redTimeLeft,numFrontVeh,arrivelTime1,arrivelTime2]
            feature1.extend(featrue2[i])
            analy1.append(feature1)

    df = pd.DataFrame(analy1,columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
                                      'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])
    df.to_csv('./data/step4b_analy1_SSVP_level7.csv',index= False)
    
    
    
    
    
    
    
    
    
    
    
###################################################################################################\
##########################################################################################################
'''B. 图1 不同标记下的最小速度分布,获得一般参数下的预测值'''


if 1:
    #columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
    #                                      'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])
    
    df = pd.read_csv('./data/step4b_analy1_SSVP_level7.csv')#来自于step4B

    stt1 = [0]*9 #[60,120] #一般参数下，统计不同标记下，不同最大速度下（建议速度下），实际最小速度分布
    stt2 = [0]*9 #[50,60]
    stt3 = [0]*9 #[40,50]
    stt4 = [0]*9 #[30,40]
    stt5 = [0]*9 #[20,30]  
    
    for j in range(9):
        originOutput = j
        dataVPTmp1 = df[df['originOutput']==originOutput]
    
    
        dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 60/3.6) & (dataVPTmp1['maxSpeed0'] < 120/3.6)]
        stt1[j] = dataVPTmp2['minSpeed0']

        dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 50/3.6) & (dataVPTmp1['maxSpeed0'] < 60/3.6)]
        stt2[j] = dataVPTmp2['minSpeed0']

        dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 40/3.6) & (dataVPTmp1['maxSpeed0'] < 50/3.6)]
        stt3[j] = dataVPTmp2['minSpeed0']

        dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 30/3.6) & (dataVPTmp1['maxSpeed0'] < 40/3.6)]
        stt4[j] = dataVPTmp2['minSpeed0']

        dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 20/3.6) & (dataVPTmp1['maxSpeed0'] < 30/3.6)]
        stt5[j] = dataVPTmp2['minSpeed0']


    for i in range(9):

        strTmp = 'label-%d' %i
        plt.title(strTmp)
        maxSpeedLabelsTmp = ['(50,60)','(40,50)','(30,40)','(20,30)','(10,20)']
        plt.boxplot([stt2[i],stt3[i],stt4[i],stt5[i],stt6[i]],showfliers=False,labels =maxSpeedLabelsTmp)

        strTmp = './data/Step4Bpy_label%d_maxSpeed.png' %i
        plt.savefig(strTmp)  # 保存为png格式图片
        plt.close() 

###################################################################################################\
###################################################################################################   
    
    
    
    
    
'''
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 40/3.6) & (dataVPTmp1['minSpeed0'] > 40/3.6)]
    analy[yKerasSumoFlag][4] = analy[yKerasSumoFlag][4]+dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 30/3.6) & (dataVPTmp1['minSpeed0'] > 30/3.6)]
    dataVPTmp2 = dataVPTmp2[(dataVPTmp2['maxSpeed0'] < 40/3.6 )& (dataVPTmp2['minSpeed0'] < 40/3.6)]
    analy[yKerasSumoFlag][3] = analy[yKerasSumoFlag][3]+dataVPTmp2.shape[0]
    
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 20/3.6) & (dataVPTmp1['minSpeed0'] > 20/3.6)]
    dataVPTmp2 = dataVPTmp2[(dataVPTmp2['maxSpeed0'] < 30/3.6) & (dataVPTmp2['minSpeed0'] < 30/3.6)]
    analy[yKerasSumoFlag][2] = analy[yKerasSumoFlag][2]+dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 15/3.6 )& (dataVPTmp1['minSpeed0'] > 15/3.6)]
    dataVPTmp2 = dataVPTmp2[(dataVPTmp2['maxSpeed0'] < 20/3.6 )& (dataVPTmp2['minSpeed0'] < 20/3.6)]
    analy[yKerasSumoFlag][1] = analy[yKerasSumoFlag][1]+dataVPTmp2.shape[0]
'''
    
'''    
    #analy1 = np.zeros((nSamples,9,9))
    
    analy1[j,0] = originOutput
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 5/3.6)]
    analy1[j,1] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 10/3.6) & (dataVPTmp1['minSpeed0'] >5/3.6)]
    analy1[j,2] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 15/3.6) & (dataVPTmp1['minSpeed0'] >10/3.6)]
    analy1[j,3] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 20/3.6) & (dataVPTmp1['minSpeed0'] >15/3.6)]
    analy1[j,4] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 25/3.6) & (dataVPTmp1['minSpeed0'] >20/3.6)]
    analy1[j,5] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 30/3.6) & (dataVPTmp1['minSpeed0'] >25/3.6)]
    analy1[j,6] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 35/3.6) & (dataVPTmp1['minSpeed0'] >30/3.6)]
    analy1[j,7] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 40/3.6) & (dataVPTmp1['minSpeed0'] >35/3.6)]
    analy1[j,8] = dataVPTmp2.shape[0]
    
    dataVPTmp2 = dataVPTmp1[(dataVPTmp1['minSpeed0'] < 120/3.6) & (dataVPTmp1['minSpeed0'] >40/3.6)]
    analy1[j,9] = dataVPTmp2.shape[0]
    
step4bAnaly1=pd.DataFrame(analy1,columns=['o1','s0','s1','s2','s3','s4','s5','s6','s7','s8'])    
    
fs = "./data/step4bAnaly1_%d.csv" %level
step4bAnaly1.to_csv(fs,index= False)    

step4bAnaly1 = pd.read_csv('./data/step4bAnaly1_7.csv', sep=',')
for i in range(9):
    
    t1 = step4bAnaly1[step4bAnaly1['o1'] == i]
    s = 's%d' %i
    
    plt.boxplot([t1['o1'],t1[s]],showfliers=False,labels=['o1',s])
    plt.title(s)
    strTmp = './data/Step4Bpy_s%d.png' %i
    plt.savefig(strTmp)  # 保存为png格式图片
    plt.close()    
    
      

step4bAnaly1 = pd.read_csv('./data/step4bAnaly1_7.csv', sep=',')

t0 = step4bAnaly1[step4bAnaly1['o1'] == 0]
t0 =  t0['s0']
t1 = step4bAnaly1[step4bAnaly1['o1'] == 1]
t1 =  t1['s1']
t2 = step4bAnaly1[step4bAnaly1['o1'] == 2]
t2 =  t2['s2']
t3 = step4bAnaly1[step4bAnaly1['o1'] == 3]
t3 =  t3['s3']
t4 = step4bAnaly1[step4bAnaly1['o1'] == 4]
t4 =  t4['s4']
t5 = step4bAnaly1[step4bAnaly1['o1'] == 5]
t5 =  t5['s5']
t6 = step4bAnaly1[step4bAnaly1['o1'] == 6]
t6 =  t6['s6']
t7 = step4bAnaly1[step4bAnaly1['o1'] == 7]
t7 =  t7['s7']
t8 = step4bAnaly1[step4bAnaly1['o1'] == 8]
t8 =  t8['s8']
plt.boxplot([t0,t1,t2,t3,t4,t5,t6,t7,t8],showfliers=False,labels=['0','1','2','3','4','5','6','7','8'])
strTmp = './data/Step4Bpy_tmp1.png'
plt.savefig(strTmp)  # 保存为png格式图片
plt.close()             

'''

sys.stdout=sys.__stdout__ 