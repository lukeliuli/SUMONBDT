##主程序开始######################################################################################################################
print("0.主程序开始，建立多层嵌套决策树模型，3080ti的GPU是AMD2400CPU 运算速度100倍")
print("0.这是简化程序，原始带有更多测试和原始模型的程序在mainTestCSVMLP3(hmcnf_keras).ipynb")
print("程序编号为0，注意这是mainSimpleStep0为简化的步骤0程序")
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
from imblearn.under_sampling import RandomUnderSampler
import pickle 
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes

from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

import sys
import argparse

from sklearn.model_selection import train_test_split

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
    '''
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
    #smote
    hierarchy = [2,3,4,5,6,7,8,9]
                      #0                 #1            #2          #3             #4        #5  #6      #7                 
    labelDict = {"0":["0",               "0",          "0",       "0",          "0",     "0",  "0",    "0"],\
                  "1":["12345678",       "1234567",    "123456",  "12345",      "1234",  "123","12",   "1"],\
                  "2":["12345678",       "1234567",    "123456",  "12345",      "1234",  "123","12",   "2"],\
                  "3":["12345678",       "1234567",    "123456",  "12345",      "1234",  "123","3",    "3"],\
                 "4":["12345678",        "1234567",    "123456",  "12345" ,     "1234",  "4",  "4",    "4"],\
                 "5":["12345678",        "1234567",    "123456",   "12345",      "5",     "5",  "5",    "5"],\
                 "6":["12345678",        "1234567",    "123456",  "6",         "6",     "6",  "6",    "6"],\
                 "7":["12345678",        "1234567",     "7",       "7",         "7",     "7",  "7",    "7"],\
                 "8":["12345678",         "8",          "8" ,      "8",         "8",     "8",  "8",    "8"],\
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


############################################################################
####HMCM-F ,层次模型，发现hmcn-f训练效果很差，所以采用分离式
###每一层的识别模型都是4层模型
def sepHier1(x,yOneHot,num_labels,saveName,levelIndex,numLayers,numEpochs = 10,srelu_size = 256,dropout_rate = 0.05):
    
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




def minSpeed2Tag(minSpeed):
    
    if minSpeed >40/3.6:
        speedFlag = 8
    if minSpeed <= 40/3.6 and minSpeed > 35/3.6:
        speedFlag = 7
    if minSpeed <= 35/3.6 and minSpeed > 30/3.6:
        speedFlag = 6
    if minSpeed <= 30/3.6 and minSpeed > 25/3.6:
        speedFlag = 5
    if minSpeed <= 25/3.6 and minSpeed > 20/3.6:
        speedFlag = 4
    if minSpeed <= 20/3.6 and minSpeed > 15/3.6:
        speedFlag = 3
    if minSpeed <= 15/3.6 and minSpeed > 10/3.6:
        speedFlag = 2
    if minSpeed <= 10/3.6 and minSpeed > 5/3.6:
        speedFlag = 1
    if minSpeed <= 5/3.6:
        speedFlag = 0
    return speedFlag


########################################################################################################################
########################################################################################################################
########################################################################################################################

def main():
    #python3 mainSimpleStep0.py --numEpochs 10 --trainOrNot 1 --testOrNot 1
    parser = argparse.ArgumentParser(description="step0")
    parser.add_argument('-np','--numEpochs', default=1000, type=int,help='分离式模型每层训练次数')
    parser.add_argument('-trn','--trainOrNot', default=1,type=int,help='训练吗')
    parser.add_argument('-ten','--testOrNot', default=1,type=int,help='测试吗')
    parser.add_argument('-tr','--testRatio', default = 0.9,type=float,help='测试集比例')
    args = parser.parse_args()
    numEpochs = args.numEpochs
    trainOrNot =  args.trainOrNot
    testRatio =  args.testRatio
    testOrNot = args.testOrNot


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
    print("\n2slot的数据列表为：headName2SlotXY96\n",headName2SlotXY96)
    ########################################################################################################
    '''将输出重定向到文件'''

    fs1 = open('step0_printlog.txt', 'w+')
    sys.stdout = fs1  # 将输出重定向到文件


    ########################################################################################################################
    print("0.主程序开始, 建立多层嵌套决策树模型,3080ti的GPU是AMD2400CPU 运算速度100倍")
    np.random.seed(42)
    tf.random.set_seed(42)


 
    ########################################################################################################################    
    ########################################################################################################################

    print("读取France数据并且把数据进行onehot处理")

    #file1 = "../trainData/france_0_allSamples1.csv"
    file1 = "../trainData/france_0_allSamples1_2slot.csv"
    xyDataTmp = pd.read_csv(file1)
    print("\n2slot的数据列表名为：headName2SlotXY96\n")
    #print(xyDataTmp.info())

    ##########################################################
    ##########################################################

    print("根据一些基本规则，需要把数据库中的一些明显不符合逻辑的数据清楚")
    #1.当前速度已经为0了，输出标志大于0
    #tmp1  = (xyDataTmp.iloc[:,3] == 0) & (xyDataTmp.iloc[:,-1] > 0)
    #xyDataTmp =xyDataTmp[tmp1 == False]
    h,w =  xyDataTmp.shape
    print(xyDataTmp.shape)
    droplist = []
    for i in range(h):
        speed = xyDataTmp.iloc[i,3].item()
        tag1 = minSpeed2Tag(speed)
        tag2 = xyDataTmp.iloc[i,-1].item()
        if tag1<tag2 or (tag1 == 0 and tag2>0):
            droplist.extend([i])

    #print(droplist)
    #print(max(droplist))
    xyDataTmp= xyDataTmp.drop(labels=droplist,axis=0)





    ##########################################################
    ##########################################################
    print("根据一些基本规则，数据处理，包括把laneID设为0，把vehid去掉")
    print("\n去掉vehID,minSpeedFlag的2slot的X数据列表为：headName2SlotX94\n")
    xyData = np.array(xyDataTmp)
    h,w = xyData.shape

    #x = xyData[:,1:23]#简单处理与SUMO数据库一致
    x0rigin = xyData[:,1:w-1]#用所有的数据,第0列为vehID,不要，
    y0rigin  = xyData[:,w-1]

    x0rigin[:,6] = [string2int(inputString) for inputString in x0rigin[:,6] ]#字符串vehLaneID 变为整数
    x0rigin[:,6] = [0 for inputString in x0rigin[:,6] ]#字符串vehLaneID 变为整数
    x0rigin =x0rigin.astype(np.float32)#GPU 加这个
    y0rigin =y0rigin.astype(np.int64)#GPU 加这个

    if 0:
        ros = RandomOverSampler(random_state=0)
        x0,y0= ros.fit_resample(x0rigin , y0rigin)#对数据不平衡进行处理，保证样本数一致

        print("数据X0的长度为94,x0.shape:",x0.shape,"y0.shape:",y0.shape,"y0.type:", type(y0) )

    if 0:
        ros =  RandomUnderSampler(random_state=0)
        x0,y0= ros.fit_resample(x0rigin , y0rigin)#对数据不平衡进行处理，保证样本数一致
        print("数据X0的长度为94,x0.shape:",x0.shape,"y0.shape:",y0.shape,"y0.type:", type(y0) )

    if 1:    
        smo = SMOTE(random_state=42)
        x0,y0 = smo.fit_sample(x0rigin , y0rigin)
        print("数据X0的长度为94,x0.shape:",x0.shape,"y0.shape:",y0.shape,"y0.type:", type(y0) )

    x0=x0.astype(np.float32)#GPU 加这个
    y0=y0.astype(np.int64)#GPU 加这个

    del xyDataTmp #节省内存
    del xyData #节省内存



  
    ########################################################################################################################




    if trainOrNot == 1:# 训练多级模型
        print("训练分离式多级模型")

        #准备字典，用于保存训练后的数据"
        xFloors=  dict()
        yFloors =  dict()
        xTestFloors =dict()
        yTestFloors = dict()
        modSaveNameFloors =dict()
        encLevels= dict()
        yKerasFloors = dict()
        x=x0
        y=y0
        x=x.astype(np.float32)#GPU 加这个
        y=y.astype(np.int64)#GPU 加这个
        print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
        print(y)

        #hierarchy = [2,4,6,8,9]
        #hierarchy = [2,3,4,5,6,7,8,9]
        yH1,hierarchy = convertY2Hieral(y)


        x_train, x_test, y_train, y_test = train_test_split(x, yH1, test_size=testRatio, random_state=0)

        nSamples,nFeatures =  x_train.shape


        #numEpochs =10 #1500/60/60*5 = 2hour #17000正确率过高


        for i in range(len(hierarchy)):   
            print("\n\n levelIndex",i,"nSamples,nFeatures",x_train.shape)
            levelIndex = i
            numLayers = 4
            enc = OneHotEncoder()
            nSamples,nFeatures =  x_train.shape


            yCurLayer1 = [t1[i] for t1 in y_train]

            yCurLayer1 = np.array(yCurLayer1)
            print("yCurLayer1.shape:",yCurLayer1.shape)

            yCurLayer1= yCurLayer1.reshape(nSamples,-1)
            enc.fit(yCurLayer1)

            yOneHot=enc.transform(yCurLayer1).toarray()
            print(enc.categories_,enc.get_feature_names())
            print(yOneHot[:1])


            num_labels = hierarchy[i] 
            print("num_labels:", num_labels)
            saveName = "../trainedModes/modelSep-9level%d-%dlayer-2slots-gpu1.h5" %(i,numLayers)
            #saveName = "../trainedModes/modelSep-2level%d-%dlayer-2slots-gpu1.h5" %(i,numLayers)#基于拥堵定义的2层结构
            print(saveName)
            sepHier1(x_train,yOneHot,num_labels,saveName,levelIndex,numLayers,numEpochs)

            encLevels[str(i)] = enc
            xFloors[str(i)] = x_train
            yFloors[str(i)] = yCurLayer1


            nSamplesTest,nFeaturesT =  x_test.shape
            yCurLayerTest = [t1[i] for t1 in y_test]
            yCurLayerTest = np.array(yCurLayerTest)
            yCurLayerTest= yCurLayerTest.reshape(nSamplesTest,-1)

            xTestFloors[str(i)] = x_test
            yTestFloors[str(i)] = yCurLayerTest
            modSaveNameFloors[str(i)] = saveName

        #######保存为pickle文件,用于后期的SUMO和数据分析
        #fpk=open('samples1.pkf','wb+')  
        #pickle.dump([xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors],fpk)  
        #fpk.close() 

        fpk=open('sepTrainedsSamplesAll1.pkf','wb+')  
        pickle.dump([xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors],fpk)  
        fpk.close() 

    ########################################################################################################################    
    ########################################################################################################################
    #####用现有训练模型进行预测

    if testOrNot == 1:

        fpk=open('sepTrainedsSamplesAll1.pkf','rb') 
        [xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors]=pickle.load(fpk)  
        fpk.close()  


        yKerasFloors = dict()

        for i in range(len(hierarchy)):
                levelIndex = i
                #x = xFloors[str(i)]
                #yCurLayer1 =  yFloors[str(i)]

                x = xTestFloors[str(i)]
                yCurLayer1 =  yTestFloors[str(i)]

                saveName =  modSaveNameFloors[str(i)] 
                enc = encLevels[str(i)]
                yOneHot=enc.transform(yCurLayer1).toarray()
                yPredict=getKerasResnetRVL(x,enc,saveName)
                print("分离式多层识别结果:第%d层\n" %i)
                mat1num = confusion_matrix(yCurLayer1,yPredict)
                print(mat1num)
                mat2acc = confusion_matrix(yCurLayer1,yPredict,normalize='pred')  
                print(np.around(mat2acc , decimals=3))
                yKerasFloors[str(i)] =  yPredict

                df = pd.DataFrame(np.around(mat2acc , decimals=3))
                fs = "./data/step0_test_mat2acc%d.csv" %i
                df.to_csv(fs,index= False, header= False)

        fpk=open('sepTestRVLSamples1.pkf','wb+')         
        pickle.dump([xFloors,yFloors,modSaveNameFloors,encLevels,yKerasFloors,xTestFloors,yTestFloors],fpk)  
        fpk.close() 

 
      
if __name__=="__main__":
    main()