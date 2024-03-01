
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import pickle 
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import random

import sys 
from sumoSimOnce1 import simOnce

from multiprocessing import Process
from datetime import datetime
import math
import time


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
print("sumoSimDataLevel7.csv，iloc[index] 与xOriginSumoAdded也就是xlowpra2 iloc对应")




############################################################################################################
############################################################################################################
############################################################################################################
print("程序4D: 在论文中体现，实现优化参数后，验证不停车过路口规则")
print("接程序3:程序3为根据输入的SUMO模拟特征对低概率样本进行重新训练,从而获得识别正确率提升")
print("mainSimpleStep4D,需要用4B中的step4b_analy1_SSVP_level7.csv和4c中的pipeline的")


########################################################################################################
'''将输出重定向到文件'''
import sys 
fs1 = open('printlog.txt', 'w+')
sys.stdout = fs1  # 将输出重定向到文件

########################################################################################################
#######################################################################################################
##基础数据格式

level = 7

if level==7:
    dfSumoData = pd.read_csv('./data/sumoSimDataLevel7_simNum500.csv', sep=',')

    headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                                   'smv1','smv2']
    xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                            'outputAvgSpeed','smv1','smv2']
    
    #dfSimVehParams = pd.read_csv('./data/paramsVehAllLevel7_simNum500.csv', sep=',')
    
if level==2:
    
    dfSumoData = pd.read_csv('./data/sumoSimDataLevel2.csv', sep=',')

    headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3',\
                                                   'smv1','smv2']
    
    xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                   'NN0','NN1','NN2','NN3','outputAvgSpeed','smv1','smv2']
    
    #dfSimVehParams = pd.read_csv('./data/paramsVehAllLevel2.csv', sep=',')





headSimVehParams = ['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                       'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                               'minSpeed0']

    
    
#################################################################
#################################################################
'''线性，预测reacTime'''

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def rectTimeReg():
    df = pd.read_csv('./data/step4b_analy1_SSVP_level7.csv')
    numSamples = df.shape[0]
    pipelineList = [0]*9
    mseList= [0]*9
    for i in range(9):
        originOutput = i
        dataVPTmp1 = df[df['originOutput']==i]
        numSamples = dataVPTmp1.shape[0]
        dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=originOutput*5/3.6) & (dataVPTmp1['minSpeed0']<(originOutput+1)*5/3.6)] 
      
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
        
        pipelineList[i] =pipeline 
        mseList[i] =mse
        
    
    return pipelineList,mseList
    
      
    
    
def testRectTimeReg():



    '''A.图2 不同标记下的最小速度分布,获得优化参数下的预测值'''

    #columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
    #                                      'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])

    #读，识别和模拟合成在一起的样本的基本数据库,step4b_analy1_SSVP_level7.csv'
    dfSSVP = pd.read_csv('./data/step4b_analy1_SSVP_level7.csv')#来自于step4B

    #clomus = headName2SlotX94 
    strTmp = 'lowprobSamplesLevel%d.pkf' %level
    fpk=open(strTmp,'rb')   
    [xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
    print("xlowpra",xlowpra.shape)
    fpk.close()  

    pipelineList,mseList = rectTimeReg()
    predRectTime = []

    for i in range(9):
        originOutput = i

        dataLabel = dfSumoData[dfSumoData['originOutput']==originOutput]

        numSample,nFeatures = dataLabel.shape
        print("label:%d,numSamples:%d,nFeatures:%d" %(i,numSample,nFeatures))


        mse = mseList[i]
        pipeline = pipelineList[i]

        #测试用
        numSample = 100

        for j in range(numSample):
            dfTmp1 = dataLabel.iloc[j]                        
            sampleIndex =int(dfTmp1['sampleIndex'].item())

            xtmp = xlowpra[sampleIndex]##clomus = headName2SlotX94 
            vehs = xtmp[8:48]
            vehs = vehs.reshape(-1,2)
            vehs = vehs[np.where(vehs[:,0]>0)]
            vehsOthers1 = vehs[0:-1]#最后一个是目标车，不要
            numFrontVeh,tmp = vehsOthers1.shape

            redTimeLeft = xtmp[0]
            dist = xtmp[1] 
            speed = xtmp[2]
            maxSpeed = xtmp[3]
            arriveTime1 = xtmp[4]
            arriveTime2= xtmp[5]


            xTmp1  = [dist,speed,redTimeLeft,numFrontVeh,maxSpeed,arriveTime1,arriveTime2]
            x = np.array([xTmp1])
            y_pred = pipeline.predict(x)
            reacTime_mu = y_pred[0]
            reacTime_sigma = mse
            predRectTime.append([sampleIndex,i,reacTime_mu,reacTime_sigma])

    dfTmp = pd.DataFrame(predRectTime,columns=['sampleIndex','label','reacTime_mu','reacTime_sigma'])
    strtmp = "./data/step4D_predRectTime.csv"
    dfTmp.to_csv(strtmp,index= False)


##################################################################################################################################################
##################################################################################################################################################
'''只运行一个例子，numSample = 100,simNum = 50,耗时6分钟
A.图2 不同标记下的最小速度分布,获得优化参数下的预测值'''
'''
def  runMain(labelList,numRunSample1=1000000,simNum = 50,level= 7):
    
    
    strTmp = './data/printlog_step4D_labelList%s.txt' %(str(labelList))
    fs1 = open(strTmp, 'w+')
    sys.stdout = fs1  # 将输出重定向到文件

    #columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
    #                                      'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])

    #读，识别和模拟合成在一起的样本的基本数据库,step4b_analy1_SSVP_level7.csv'
    strTmp = './data/step4b_analy1_SSVP_level%d.csv' %level
    dfSSVP = pd.read_csv(strTmp)#来自于step4B

    #clomus = headName2SlotX94 
    strTmp = 'lowprobSamplesLevel%d.pkf' %level
    fpk=open(strTmp,'rb')   
    [xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
    print("xlowpra",xlowpra.shape)
    fpk.close()  

    pipelineList,mseList = rectTimeReg()
 
    
    analy3 = []
    for i in labelList:
        originOutput = i

        dataLabel = dfSumoData[dfSumoData['originOutput']==originOutput]
        numSample,nFeatures = dataLabel.shape
        print("label:%d,numSamples:%d,nFeatures:%d" %(i,numSample,nFeatures)) 

        #测试用
        numSample = min(numRunSample1,numSample)

        for j in range(numSample):
            print("\n\n label,iLoc:",i,j)
            dfTmp1 = dataLabel.iloc[j]                        
            sampleIndex = int(dfTmp1['sampleIndex'].item())
            xtmp = xlowpra[sampleIndex]##clomus = headName2SlotX94 
            xtmp = xtmp[0:48]
            vehs = xtmp[8:48]
            vehs = vehs.reshape(-1,2)
            vehs = vehs[np.where(vehs[:,0]>0)]
            vehsOthers1 = vehs[0:-1]#最后一个是目标车，不要
            numFrontVeh,tmp = vehsOthers1.shape

            redTimeLeft = xtmp[0]
            dist = xtmp[1] 
            speed = xtmp[2]
            maxSpeed = xtmp[3]
            arriveTime1 = xtmp[4]
            arriveTime2= xtmp[5]

            mse = mseList[i]
            pipeline = pipelineList[i]

            xTmp1  = [dist,speed,redTimeLeft,numFrontVeh,maxSpeed,arriveTime1,arriveTime2]
            x = np.array([xTmp1])
            y_pred = pipeline.predict(x)
            reacTime_mu = y_pred[0]
            reacTime_sigma = mse
            predRectTime = [reacTime_mu,reacTime_sigma]


            
            label= i
            counter =  j
            for k in range(5):
                maxSpeedFlag = k
                maxSpeedList =[[20/3.6,30/3.6],[30/3.6,40/3.6],[40/3.6,50/3.6],[50/3.6,60/3.6],[60/3.6,70/3.6]]
                minSpeedList1,paramsVehList,outputAvgSpeed,sumoOutputSpeedTag = simOnce(xtmp,simNum,predRectTime,maxSpeedList[maxSpeedFlag])
                if outputAvgSpeed>0:
                    analy3.append([sampleIndex,label,counter,simNum,minSpeedList1,paramsVehList,outputAvgSpeed,sumoOutputSpeedTag,maxSpeedFlag])


    df_analy3 = pd.DataFrame(analy3,columns=['sampleIndex','label','counter','simNum','minSpeedList1','paramsVehList',\
                                            'outputAvgSpeed','sumoOutputSpeedTag','maxSpeedFlag'])

    fs = "./data/step4D_simData_level%d_labelList%s.csv" %(level,str(labelList))
    df_analy3.to_csv(fs,index=False)
    
    analyz1(labelList, df_analy3)
   
###################################################################################################    
def analyz1(labelList, df_analy3):
    #labelList = range(9)
    stt =  np.zeros([9,5]).tolist()
    nonStopRatio = np.zeros([9,5]).tolist()

    
    for j in labelList:
        label = j
        dataVPTmp1 = df_analy3[df_analy3['label']==label]

        for k in range(5):
            dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeedFlag'] == k)]
            
            tmpValues2 = dataVPTmp2['minSpeedList1']
            dataTmp = []
            for ii in range(len(tmpValues2)):
                tmpV = tmpValues2.iloc[ii]
                dataTmp.extend(tmpV.tolist())
            
            stt[j][k] = dataTmp
            
 
    
               
    #labelList = range(9)
    #labelTitle = ['0','1','2','3','4','5','6','7','8','9']
    labelTitle = [str(x) for x in labelList]
    maxSpeedListTmp =['[20,30]','[30,40]','[40,50]','[50,60]','[60,70]']
    
    for j in labelList:

        strTmp = 'label-%d' %j
        plt.title(strTmp)
        plt.boxplot([stt[j][0],stt[j][1],stt[j][2],stt[j][3],stt[j][4]],showfliers=False,labels =maxSpeedListTmp)
        strTmp = './data/Step4Dpy_label%d_maxSpeed.png' %j
        plt.savefig(strTmp)  # 保存为png格式图片
        plt.close()  
        

    for i in labelList:
        for j in range(5):

            maxSpeedList =[[20,30],[30,40],[40,50],[50,60],[60,70]]
            s1,s2 =  maxSpeedList[j]

            minSpeedList = stt[i][j]
            minSpeedList1 = [num for num in  minSpeedList if num > s1/3.6 and num <s2/3.6]
            
           
            nonStopRatio[i][j] = len(minSpeedList1)/len(minSpeedList) *100

    strTmp = './data/stage4d_minSpeedList_ratio_labelList%s.pkf' %str(labelList)
    fpk=open(strTmp,'wb+')  
    pickle.dump([nonStopRatio,stt],fpk)
    fpk.close()    
    nonStopRatio = np.round(nonStopRatio,4)
    print(nonStopRatio)
    
    dfTmp = pd.DataFrame(nonStopRatio,columns=maxSpeedListTmp)
    strtmp = "./data/step4D_nonStopRatio_labelList%s.csv" %str(labelList)
    dfTmp.to_csv(strtmp,index= False)
    

    
###################################################################################################
###################################################################################################
#labelList = range(1),numSample1=20,simNum = 50,耗时6分钟
#labelList = range(9),numSample1=20,simNum = 10,耗时10分钟
#labelList = range(9),numSample1=20,simNum = 30,耗时25分钟
#labelList = range(9),numSample1=200,simNum = 100,耗时???25*30分钟
#multiProcess,labelList = range(9),numSample1=20,simNum = 10,耗时2min20sec,6核12线程
#multiProcess,labelList = range(9),numSample1=200,simNum = 50,耗时?? 2min20sec *50=97min

from multiprocessing import Process
from datetime import datetime
import math
import time

'''

'''
def mainMultiprocessing1():
    
    timeST = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    numRunSample=200
    simNum = 50
   
    if 0: 
        
        sectionNum = 6
        sectionProcess = dict()
        labelList = dict()
        labelList = [[0],[1,2],[3,4],[5,6],[7],[8]]
        level = 7
     
        
        for i in np.arange(sectionNum):
            sectionProcess[i] = Process(target=runMain, args=(labelList[i], numRunSample,simNum,level ))
            
        for i in np.arange(sectionNum):
            sectionProcess[i].start()
            
        for i in np.arange(sectionNum):
            sectionProcess[i].join()
       
   
    
    if 1:
        sectionNum = 6
        labelListA = [[0],[1,2],[3,4],[5,6],[7],[8]]
        analy = np.zeros([9,5]).tolist()
        for i in np.arange(sectionNum):
            strTmp = './data/step4D_numSample1=200_simNum = 50/step4D_nonStopRatio_labelList%s.csv' %str(labelListA[i])
            dfData = pd.read_csv(strTmp, sep=',')
            for j in labelListA[i]:
                tmpData =dfData.iloc[j]
                analy[j][0] = tmpData[0]
                analy[j][1] = tmpData[1]
                analy[j][2] = tmpData[2]
                analy[j][3] = tmpData[3]
                analy[j][4] = tmpData[4]
         
        maxSpeedListTmp =['[20,30]','[30,40]','[40,50]','[50,60]','[60,70]']
        dfTmp = pd.DataFrame(analy,columns=maxSpeedListTmp)
        strtmp = "./data/step4D_nonStopRatio_all.csv" 
        dfTmp.to_csv(strtmp,index= False)
        
    timeED = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    print("start:%s" %timeST)
    print("end:%s" %timeED)
    
'''    
###################################################################################################
def analyMaxSpeed(df_analy3,labelList):
    stt =  (np.zeros([9,5])).tolist()
  
    nonStopRatio = np.zeros([9,5]).tolist()

    
    for j in labelList:
        label = j
        dataVPTmp1 = df_analy3[df_analy3['label']==label]

        for k in range(5):
            dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeedFlag'] == k)]
            
            tmpValues2 = dataVPTmp2['minSpeedList1']
            
            dataTmp = []
            for ii in range(len(tmpValues2)):
                tmpV = tmpValues2.iloc[ii]
                dataTmp.extend(tmpV)
            
            stt[j][k] = dataTmp
            
 
    
               
    #labelList = range(9)
    #labelTitle = ['0','1','2','3','4','5','6','7','8','9']
    labelTitle = [str(x) for x in labelList]
    maxSpeedListTmp =['[20,30]','[30,40]','[40,50]','[50,60]','[60,70]']
    
    for j in labelList:

        strTmp = 'label-%d' %j
        #plt.title(strTmp)
        #plt.boxplot([stt[j][0],stt[j][1],stt[j][2],stt[j][3],stt[j][4]],showfliers=False,labels=maxSpeedListTmp)
        #strTmp = './data/Step4Dpy_label%d_maxSpeed.png' %j
        #plt.savefig(strTmp)  # 保存为png格式图片
        #plt.close()  
        

    for i in labelList:
        for j in range(5):

            maxSpeedList =[[20,30],[30,40],[40,50],[50,60],[60,70]]
            s1,s2 =  maxSpeedList[j]

            minSpeedList = stt[i][j]
            minSpeedList1 = [num for num in  minSpeedList if num > s1/3.6 and num <s2/3.6]
            
           
            nonStopRatio[i][j] = len(minSpeedList1)/len(minSpeedList) *100

    strTmp = './data/stage4D_minSpeedList.pkf'
    fpk=open(strTmp,'wb+')  
    pickle.dump([nonStopRatio,stt],fpk)
    fpk.close()    
    nonStopRatio = np.round(nonStopRatio,4)
    print(nonStopRatio)
    
    dfTmp = pd.DataFrame(nonStopRatio,columns=maxSpeedListTmp)
    strtmp = "./data/step4D_nonStopRatio.csv"
    dfTmp.to_csv(strtmp,index= False)
    
    
    
###################################################################################################

def  runMain(labelList, sectionValue,sectionName,numRunSample =60,simNum = 10,level= 7):
    
    
    strTmp = './data/printlog_step4D_%s.txt' %(sectionName)
    fs1 = open(strTmp, 'w+')
    sys.stdout = fs1  # 将输出重定向到文件

    #columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
    #                                      'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])

    #读，识别和模拟合成在一起的样本的基本数据库,step4b_analy1_SSVP_level7.csv'
    strTmp = './data/step4b_analy1_SSVP_level%d.csv' %level
    dfSSVP = pd.read_csv(strTmp)#来自于step4B

    #clomus = headName2SlotX94 
    strTmp = 'lowprobSamplesLevel%d.pkf' %level
    fpk=open(strTmp,'rb')   
    [xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
    print("xlowpra",xlowpra.shape)
    fpk.close()  

    pipelineList,mseList = rectTimeReg()
 
    
    analy3 = []
    for i in labelList:
        originOutput = i

        dataLabel = dfSumoData[dfSumoData['originOutput']==originOutput]
        numSample,nFeatures = dataLabel.shape
        print("label:%d,numSamples:%d,nFeatures:%d" %(i,numSample,nFeatures)) 

        #测试用
        numSamples = min(numRunSample,numSample)
        
        stt = round(numSamples*sectionValue[0])
        end = round(numSamples*sectionValue[1])
        jTmpIndex = np.arange(stt,end)

        strLog = "start:%d,end:%d" %(stt,end)
        print(strLog)

        for j in jTmpIndex:
            print("\n\n label,iLoc:",i,j)
            dfTmp1 = dataLabel.iloc[j]                        
            sampleIndex = int(dfTmp1['sampleIndex'].item())
            xtmp = xlowpra[sampleIndex]##clomus = headName2SlotX94 
            xtmp = xtmp[0:48]
            vehs = xtmp[8:48]
            vehs = vehs.reshape(-1,2)
            vehs = vehs[np.where(vehs[:,0]>0)]
            vehsOthers1 = vehs[0:-1]#最后一个是目标车，不要
            numFrontVeh,tmp = vehsOthers1.shape

            redTimeLeft = xtmp[0]
            dist = xtmp[1] 
            speed = xtmp[2]
            maxSpeed = xtmp[3]
            arriveTime1 = xtmp[4]
            arriveTime2= xtmp[5]

            mse = mseList[i]
            pipeline = pipelineList[i]

            xTmp1  = [dist,speed,redTimeLeft,numFrontVeh,maxSpeed,arriveTime1,arriveTime2]
            x = np.array([xTmp1])
            y_pred = pipeline.predict(x)
            reacTime_mu = y_pred[0]
            reacTime_sigma = mse
            predRectTime = [reacTime_mu,reacTime_sigma]


            
            label= i
            counter =  j
            for k in range(5):
                maxSpeedFlag = k
                maxSpeedList =[[20/3.6,30/3.6],[30/3.6,40/3.6],[40/3.6,50/3.6],[50/3.6,60/3.6],[60/3.6,70/3.6]]
                minSpeedList1,paramsVehList,outputAvgSpeed,sumoOutputSpeedTag = simOnce(xtmp,simNum,predRectTime,maxSpeedList[maxSpeedFlag])
                if outputAvgSpeed>0:
                    analy3.append([sampleIndex,label,counter,simNum,minSpeedList1,paramsVehList,outputAvgSpeed,sumoOutputSpeedTag,maxSpeedFlag])


    df_analy3 = pd.DataFrame(analy3,columns=['sampleIndex','label','counter','simNum','minSpeedList1','paramsVehList',\
                                            'outputAvgSpeed','sumoOutputSpeedTag','maxSpeedFlag'])

    #因为输入里面有Flaot,List，保存为csv会导致'minSpeedList1'读取为obj,和string,而不是float list
    #fs = './data/step4D_simData_level%d_%s.csv' %(level,sectionName)
    #df_analy3.to_csv(fs,index=False)
    
    strTmp ='./data/step4D_simData_level%d_%s.pkf' %(level,sectionName)
    fpk=open(strTmp,'wb+')  
    pickle.dump([df_analy3,sectionName],fpk)
    fpk.close()    
   
   
###################################################################################################
#numSamples = 18,simNum = 2 ，section = 6,amd2400,34sec
#numSamples = 18,simNum = 5 ，section = 6,amd2400,1min45sec
#numSamples = 180,simNum = 5 ，section = 6,amd2400,9min20sec
#numSamples = 180,simNum = 5 ，section = 12,amd2400,7min34sec

def mainMultiprocessing2():
    
    timeST = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    sectionNum = 12
    sectionProcess = dict()
    labelList = list(range(9))

    level = 7
    numSamples = 180
    simNum = 5
   
    if 1: 
        
        for i in np.arange(sectionNum): 
            sectionValue = [i/sectionNum,(i+1)/sectionNum]
            sectionName = "section%d[%.3f_%.3f]" %(i,sectionValue[0],sectionValue[1])
            sectionProcess[i] = Process(target=runMain, args=(labelList, sectionValue,sectionName,numSamples,simNum,level))
            
        for i in np.arange(sectionNum):
            sectionProcess[i].start()
            
        for i in np.arange(sectionNum):
            sectionProcess[i].join()
       
   
    
    if 1:#将平行数据转为数据整合一个文件
        dfData = pd.DataFrame()
        for i in np.arange(sectionNum):
            sectionValue = [i/sectionNum,(i+1)/sectionNum]
            sectionName = "section%d[%.3f_%.3f]" %(i,sectionValue[0],sectionValue[1])
            
            strTmp = './data/step4D_simData_level%d_%s.pkf' %(level,sectionName)
            fpk=open(strTmp,'rb')  
            [df_section,name] = pickle.load(fpk)
            fpk.close()  
            dfData = pd.concat([dfData,df_section])
        
         
        analyMaxSpeed(dfData,labelList)    
    
    timeED = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    print("start:%s" %timeST)
    print("end:%s" %timeED)    
    
###################################################################################################
###################################################################################################

mainMultiprocessing2()