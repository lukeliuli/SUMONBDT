
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
print("程序4C: 在论文中体现，实现蒙特卡洛模拟的参数估计")
print("mainSimpleStep4C注意只能接mainSimpleStep3的输出结果")


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



    
########################################################################################################   
########################################################################################################

    
########################################################################################################    
########################################################################################################
'''A.相似度分析'''

'''将输出重定向到文件'''
import sys 
fs1 = open('printlog_step4C.txt', 'w+')
sys.stdout = fs1  # 将输出重定向到文件


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def main(): 
    

    df = pd.read_csv('./data/step4b_analy1_SSVP_level7.csv')


    numSamples = df.shape[0]
    print(numSamples)


    for i in range(9):
        originOutput = i
        dataVPTmp1 = df[df['originOutput']==i]
        numSamples = dataVPTmp1.shape[0]
        dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=originOutput*5/3.6) & (dataVPTmp1['minSpeed0']<(originOutput+1)*5/3.6)] 
        methond2(dataVPTmp1,i)
        print(i)
 





#################################################################
#################################################################
''' 线性回归'''    
def methond2(dataVPTmp1,label): 
    x = dataVPTmp1[['dist','speed','redTimeLeft','numFrontVeh','maxSpeed','arriveTime1', 'arriveTime2']]
    x = x.to_numpy()
    y = dataVPTmp1['reacTime'].to_numpy()
    
    # 构建pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])

    # 训练模型
    pipeline.fit(x, y)

    # 预测结果
    y_pred = pipeline.predict(x)

    # 均方误差
    mse = mean_squared_error(y, y_pred)
    print(y[0:10])
    print(y_pred[0:10])
    print('MSE: %.3f' % mse)
       
    
    # 获取特征重要性
    importance = pipeline.named_steps['reg'].coef_

    # 将特征重要性与对应特征名对应
    feature_names = ['dist','speed','redTimeLeft','numFrontVeh','maxSpeed','arriveTime1', 'arriveTime2']
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # 绘制水平条形图
    plt.figure(figsize=(10,12))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('Feature importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    strTmp = './data/Step4C_reg_%d.png' %label
    plt.savefig(strTmp)  # 保存为png格式图片
    plt.close()

        
        
    strTmp = './data/stage4c_fit1_label%d.pkf' %label
    fpk=open(strTmp,'wb+')  
    pickle.dump([dataVPTmp1,x,y,pipeline,mse],fpk)  
    fpk.close()     
    
    
    
    
#################################################################
#################################################################
''' 近似度+多项式特征+线性回归'''
def methond1(dataVPTmp1):
    simil_x2 = dataVPTmp1[['dist','speed','redTimeLeft','numFrontVeh','maxSpeed','arriveTime1', 'arriveTime2']]
    simil_x2 = simil_x2.to_numpy()
    
    
    
    
    
    numSamples = simil_x2.shape[0]
    print('numSamples',numSamples)
    print('simil_x2.shape',simil_x2.shape)
  
    analy2 = []
    numSamples = 1000
    for j in range(numSamples):
        tmp1 = dataVPTmp1.iloc[j]
        sampleIndex = tmp1['sampleIndex']
        simil_x1 =  tmp1[['dist','speed','redTimeLeft','numFrontVeh','maxSpeed','arriveTime1', 'arriveTime2']]
        simil_x1 = simil_x1.to_numpy()
        simil_x1 = simil_x1.reshape(1, -1)
        s2 = cosine_similarity(simil_x1,simil_x2)

        tmp = s2>0.95
        tmp = tmp[0]
        rvl = simil_x2[tmp,:]
        dataVPTmp2 = dataVPTmp1[tmp]
        #print("i,j:",i,j)
        #print(np.round(simil_x1,2))
        for k in range(rvl.shape[0]):
            if  simil_x1[0][2] == rvl[k,2] and simil_x1[0][3] == rvl[k,3]:
                analy2.append(dataVPTmp2.iloc[k].tolist())
                
    #analy2 = np.round(analy2,2)
    #print(analy2) 
    dfTmp = pd.DataFrame(analy2,columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
                              'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])
    strtmp = "./data/step4c_nn_sim%d_level%d_label%d.csv" %(500,level,i)
    dfTmp.to_csv(strtmp,index= False)
    
    dfTmp2 = dfTmp[['dist','speed','redTimeLeft','numFrontVeh','maxSpeed','arriveTime1', 'arriveTime2']]
    x = dfTmp2.to_numpy()
    y = dfTmp['reacTime'].to_numpy()
                
                
    # 构建pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=1, include_bias=True)),
        ('reg', LinearRegression())
    ])

    # 训练模型
    pipeline.fit(x, y)

    # 预测结果
    y_pred = pipeline.predict(x)

    # 均方误差
    print(y[0:10])
    print(y_pred[0:10])
    mse = mean_squared_error(y, y_pred)

    # R2值
    r2 = r2_score(y, y_pred)

    print('MSE: %.3f' % mse)
    print('R2 score: %.3f' % r2)      
    
    # 获取特征重要性
    importance = pipeline.named_steps['reg'].coef_

    # 将特征重要性与对应特征名对应
    feature_names = pipeline.named_steps['poly'].get_feature_names(dfTmp2.columns)
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # 绘制水平条形图
    plt.figure(figsize=(10,12))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('Feature importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    strTmp = './data/Step4C_reg_%d.png' %i
    plt.savefig(strTmp)  # 保存为png格式图片
    plt.close()

        
        
    strTmp = './data/stage4c_fit1_label%d.pkf' %i
    fpk=open(strTmp,'wb+')  
    pickle.dump([dfTmp,x,y,pipeline,mse],fpk)  
    fpk.close()     
        
        
#################################################################
#################################################################        
        
main()