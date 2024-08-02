
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
import sys
import argparse
from numpy import percentile as pct

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

'''step2获得：step1中低概率样本中SUMO成果仿真获取的样本（有一定挑选，去掉无法仿真样本大概1%）'''
'''step3获得：将step2的样本与对应的step1的低概率样本的进行合成，并训练加强模型stage2Mode''' 



############################################################################################################
############################################################################################################
############################################################################################################
print("程序4B: 在论文中体现，实现验证不停车过路口规则,只采用一般参数下模拟的结构")
print("mainSimpleStep4B注意只能接mainSimpleStep3的输出结果")
'''step2获得：step1中低概率样本中SUMO成果仿真获取的样本（有一定挑选，去掉无法仿真样本大概1%）'''
'''step3获得：将step2的样本与对应的step1的低概率样本的进行合成，并训练加强模型stage2Mode'''    



########################################################################################################

def main():
    
    ########################################################################################################
    '''将输出重定向到文件'''
    import sys 
    fs1 = open('step4B_printlog.txt', 'w+')
    sys.stdout = fs1  # 将输出重定向到文件    
    #sys.stdout=sys.__stdout__ 

    ########################################################################################################
    #python3 mainSimpleStep4B.py --level 7 --simNum 15  --genSimDataOrNot 1 --nSamples 10000
    parser = argparse.ArgumentParser(description="step4B")
    parser.add_argument('-ll','--level', default=7,type=int,help='step3的运行第几层')
    parser.add_argument('-su','--simNum', default=15, type=int,help='step2_每个例子要仿真的次数')
    parser.add_argument('-gs','--genSimDataOrNot', default=1, type=int,help='将场景信息和仿真参数进行合成')
    parser.add_argument('-ns','--nSamples', default=10000, type=int,help='需要4B处理多少个样本')
    
    
  
    args = parser.parse_args()
    level = args.level
    simNum = args.simNum
    genSimDataOrNot = args.genSimDataOrNot
    nSamplesTry= args.nSamples
    
    '''step2获得：step1中低概率样本中SUMO成果仿真获取的样本（有一定挑选，去掉无法仿真样本大概1%）'''
    '''step3获得：将step2的样本与对应的step1的低概率样本的进行合成，并训练加强模型stage2Mode''' 
 
    '''获得step2中的SUMO模拟的数据'''
    if level==7:
        strTmp = './data/step2_sumoSimDataLevel%d_simNum%d.csv' %(level,simNum)
        dfSumoData = pd.read_csv(strTmp, sep=',')
        headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                       'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                                       'smv1','smv2']
        xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                       'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                                'outputAvgSpeed','smv1','smv2']
        
        strTmp = './data/step2_paramsVehAllLevel%d_simNum%d.csv' %(level,simNum)
        dfSimVehParams = pd.read_csv(strTmp, sep=',')

    if level==2:

        strTmp = './data/step2_sumoSimDataLevel%d_simNum%d.csv' %(level,simNum)
        dfSumoData = pd.read_csv(strTmp, sep=',')

        headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                                       'NN0','NN1','NN2','NN3',\
                                                       'smv1','smv2']

        xInputHeader = headName2SlotX94+['sumoOutputSpeedTag','kerasPredictLabel',\
                                                       'NN0','NN1','NN2','NN3','outputAvgSpeed','smv1','smv2']

        strTmp = './data/step2_paramsVehAllLevel%d_simNum%d.csv' %(level,simNum)
        dfSimVehParams = pd.read_csv(strTmp, sep=',')






    headSimVehParams = ['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                           'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                                   'minSpeed0']




    ###################################################################################################\
    ##########################################################################################################  
    '''A.获得识别和模拟合成在一起的样本的基本数据库,step4b_analy1_SSVP_level7.csv'''
    '''保证等到的样本的样本最小速度等于实际预测类别'''

    #fpk=open('stage2-LowprobSamples-addingSumoFeatures-Hierachical.pkf','rb')   
    #[xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors]=pickle.load(fpk)  
    #fpk.close()  
    
    '''step2获得：step1中低概率样本中SUMO成果仿真获取的样本（有一定挑选，去掉无法仿真样本大概1%）'''
    '''step3获得：将step2的样本与对应的step1的低概率样本的进行合成，并训练加强模型stage2Mode'''
    #step3,首先读入step2的SUMO模拟数据，(SUMO模拟数据不完整)
    #接着，获得step2的SUMO模拟数据面的样本的序号，
    #然后，按照序号获得step1里面的低概率样本的所有数据，并命名为xlowpra2(df_step2加上xlowpra2，数据就完整了）
    #'step3_modelSepStage2_level%d.pkf 里面的数据就是 SUMO 模拟数据df_step和xlowpra2的合成
    
    saveNamePKF = 'step3_modelSepStage2_level%d.pkf' %level                                                 
    fpk=open(saveNamePKF,'rb') 
    [xOriginSumoAdded,yOriginSumoAdded,saveName,enc,x_train,y_train,yKerasSumoPredict]=pickle.load(fpk)  
    fpk.close()  
    
    ''' 基于pandas用于分析输入数据
    #np.concatenate([lowproKerasStage1Input,x1_sumoOutput,x2_yKerasOutput,x3_outputListNN,x4_outputAvgSpeed,x5_sumoOutList],axis=1)#71%
    print('xOriginSumoAdded.shape:',xOriginSumoAdded.shape)
    xInputDF = pd.DataFrame(xOriginSumoAdded,columns=xInputHeader)

    fs = "./data/step4_xInput_level%d.csv" %level
    xInputDF.to_csv(fs,index= False)
    '''

    nSamples,nFeatures = xOriginSumoAdded.shape
    nSamples = min(nSamples,nSamplesTry)
    print("xOriginSumoAdded.shape:",xOriginSumoAdded.shape)
    print("dfSumoData.shape:",dfSumoData.shape)
    print("dfSimVehParams.shape:",dfSimVehParams.shape)   

    

    if genSimDataOrNot:

        analy1 =[] #将场景信息和仿真参数进行合成，并存入analy1
        
        for j in range(nSamples):
            print('#################################################################')  
            print('j:',j) 
            '''xOriginSumoAdded（也就是X），用于训练，行标号不对应sumoSimDataLevel7.csv里面的sampleIndex'''
            '''根据xOriginSumoAdded行标号，获得sumoSimDataLevel7.csv行标号对应的sampleIndex'''



            dataSD = dfSumoData.iloc[j]
            sampleIndex = dataSD['sampleIndex'].item()
            dataVP = dfSimVehParams[dfSimVehParams['sampleIndex'] == sampleIndex]
            print('sampleIndex:',sampleIndex)
            print("dataVP.shape:",dataVP.shape)


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
            
            #originOutput = int(kerasPredictLabel) #只作为level=7的测试

            print(dataSD[['originOutput','kerasPredictLabel','sumoOutputSpeedTag']])#item对于Series


            

            #分析进行
            dataVPTmp1=dataVP
            
            dataVPTmp1 = dataVPTmp1[['minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed']]
            
            #dataVPTmp1 = dataVPTmp1[(dataVPTmp1['reacTime0'] <1.2) & (dataVPTmp1['reacTime'] <1.2)]
          
            '''保证等到的样本最小速度等于实际预测类别,有问题，要改'''
            
            if level == 2:
                #kerasPredictLabel = int(kerasPredictLabel)
                #labelDict = {0:[0,5/3.6],1:[5/3.6,7*5/3.6],2:[7*5/3.6,8*5/3.6],3:[8*5/3.6,80/5.6]}
                #minmax = labelDict[kerasPredictLabel]
                #dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=minmax[0]) & (dataVPTmp1['minSpeed0']<minmax[1])] 
                
                #labelDict = {0:[0,5/3.6],123456:[5/3.6,7*5/3.6],7:[7*5/3.6,8*5/3.6],8:[8*5/3.6,80/5.6]}
                #minmax = labelDict[originOutput]
                #dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=minmax[0]) & (dataVPTmp1['minSpeed0']<minmax[1])] 
                
                pass
                
            if level == 7:  
                #dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=kerasPredictLabel*5/3.6) & (dataVPTmp1['minSpeed0']<(kerasPredictLabel+1)*5/3.6)] 
                #dataVPTmp1 = dataVPTmp1[(dataVPTmp1['minSpeed0']>=originOutput*5/3.6) & (dataVPTmp1['minSpeed0']<(originOutput+1)*5/3.6)] 
                pass
            print("dataVPTmp1.shape:",dataVPTmp1.shape)
            
            index1 = j
            index2 = sampleIndex
            feature1 = [index1,index2,originOutput,dist,speed,redTimeLeft,numFrontVeh]
            featrue2 = dataVPTmp1.values.tolist()
            for i in range(dataVPTmp1.shape[0]):
                #print(featrue2[i])
                feature1 = [index1,index2,originOutput,dist,speed,redTimeLeft,numFrontVeh,arrivelTime1,arrivelTime2]
                feature1.extend(featrue2[i])
                analy1.append(feature1)#将场景信息和仿真参数进行合成，并存入analy1

        df = pd.DataFrame(analy1,columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
                                          'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])
        
        saveName = './data/step4b_analy1_SSVP_level%d.csv' %level
        df.to_csv(saveName,index= False)











    ###################################################################################################\
    ##########################################################################################################
    '''B. 图1 不同标记下的最小速度分布,获得一般参数下的预测值'''
    '''1.等到的样本最小速度等于实际预测类别'''
    '''2.统计样本的不同类别下最大速度的分布'''
    '''3.step2,模拟时设定最大速度区间为[45,70]'''

    if 1:
        #columns=['index','sampleIndex','originOutput','dist','speed','redTimeLeft','numFrontVeh',\
        #                                      'arriveTime1', 'arriveTime2','minSpeed0','maxSpeed0','reacTime0','reacTime','maxSpeed'])

        readName = './data/step4b_analy1_SSVP_level%d.csv' %level
        df = pd.read_csv(readName)#来自于step4B

        stt1 = [0]*9 #[60,120] #一般参数下，统计不同标记下，不同最大速度下（建议速度下），实际最小速度分布
        stt2 = [0]*9 #[50,60]
        stt3 = [0]*9 #[40,50]
        #stt4 = [0]*9 #[30,40]
        #stt5 = [0]*9 #[20,30]  
        if level == 7:
            labelList = [0,1,2,3,4,5,6,7,8]
            
        if level == 2:
            labelList = [0,123456,7,8]
            
        for j in range(len(labelList)):
            originOutput = labelList[j]
            dataVPTmp1 = df[df['originOutput']==originOutput ]
            dataVPTmp1 = df[df['originOutput']==originOutput ]
            


            dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 60/3.6) & (dataVPTmp1['maxSpeed0'] < 80/3.6) & (dataVPTmp1['minSpeed0']<80/3.6)]
            stt1[j] = dataVPTmp2['minSpeed0']

            dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 50/3.6) & (dataVPTmp1['maxSpeed0'] < 60/3.6) & (dataVPTmp1['minSpeed0']<80/3.6)]
            stt2[j] = dataVPTmp2['minSpeed0']

            dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 40/3.6) & (dataVPTmp1['maxSpeed0'] < 50/3.6) & (dataVPTmp1['minSpeed0']<80/3.6)]
            stt3[j] = dataVPTmp2['minSpeed0']
            
            #dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 30/3.6) & (dataVPTmp1['maxSpeed0'] < 40/3.6)]
            #stt4[j] = dataVPTmp2['minSpeed0']

            #dataVPTmp2 = dataVPTmp1[(dataVPTmp1['maxSpeed0'] > 20/3.6) & (dataVPTmp1['maxSpeed0'] < 30/3.6)]
            #stt5[j] = dataVPTmp2['minSpeed0']

        
        record1 = np.zeros((9, 6)).tolist()
        record2 = np.zeros((9, 6)).tolist()
        
        for i in range(len(labelList)):
            originOutput = labelList[i]
            '''
            strTmp = 'label-%d' %i
            plt.title(strTmp)
            maxSpeedLabelsTmp = ['(50,60)','(40,50)','(30,40)','(20,30)']
            plt.boxplot([stt2[i]*3.6,stt3[i]*3.6,stt4[i]*3.6,stt5[i]*3.6],showfliers=True,labels =maxSpeedLabelsTmp)

            strTmp2 = './data/Step4Bpy_label%d_maxSpeed.png' %i
            plt.savefig(strTmp2)  # 保存为png格式图片
            plt.close() 
            '''
            
            dataVPTmp1 = df[df['originOutput']==originOutput]
            numSamples,nFeatures = dataVPTmp1.shape
            
            maxSpeedLabelsTmp = ['(60,80)','(50,60)','(40,50)','(30,40)','(20,30)']
            
            print(strTmp,maxSpeedLabelsTmp[0],(stt1[i].mean())*3.6,stt1[i].std())
            print(strTmp,maxSpeedLabelsTmp[1],(stt2[i].mean())*3.6,stt2[i].std())
            print(strTmp,maxSpeedLabelsTmp[2],(stt3[i].mean())*3.6,stt3[i].std())
            #print(strTmp,maxSpeedLabelsTmp[3],(stt4[i].mean())*3.6,stt4[i].std())
            #print(strTmp,maxSpeedLabelsTmp[4],(stt5[i].mean())*3.6,stt5[i].std())
         
            tmpP=[(stt1[i].mean())*3.6,stt1[i].std()]
            record1[i][0] = [round(v,3) for v in tmpP] 
            
            tmpP=[(stt2[i].mean())*3.6,stt2[i].std()]
            record1[i][1] = [round(v,3) for v in tmpP] 
            
            tmpP=[(stt3[i].mean())*3.6,stt3[i].std()]
            record1[i][2] = [round(v,3) for v in tmpP] 
            
            #record1[i][3] = [(stt4[i].mean())*3.6,stt4[i].std()]
            #record1[i][4] = [(stt5[i].mean())*3.6,stt5[i].std()]
            
            tmpP=[min(stt1[i]),pct(stt1[i],25),pct(stt1[i],50),pct(stt1[i],75),max(stt1[i])]
            record2[i][0] = [round(v*3.6,3) for v in tmpP] 
            tmpP=[min(stt2[i]),pct(stt2[i],25),pct(stt2[i],50),pct(stt2[i],75),max(stt2[i])]
            record2[i][1] = [round(v*3.6,3) for v in tmpP] 
            tmpP=[min(stt3[i]),pct(stt3[i],25),pct(stt3[i],50),pct(stt3[i],75),max(stt3[i])]
            record2[i][2] = [round(v*3.6,3) for v in tmpP] 
            
            '''
            tmpP=[min(stt4[i]),pct(stt4[i],25),pct(stt4[i],50),pct(stt4[i],75),max(stt4[i])]
            record2[i][3] = [v*3.6 for v in tmpP] 
            tmpP=[min(stt5[i]),pct(stt5[i],25),pct(stt5[i],50),pct(stt5[i],75),max(stt5[i])]
            record2[i][4] = [v*3.6 for v in tmpP] 
            '''
            
        saveName = './data/step4b_nonStopRatio_level%d.csv' %level
        record1 = pd.DataFrame(record1)
        record1.to_csv(saveName,index= False,float_format='%.2f')
        
        saveName = './data/step4b_nonStopRatio2_level%d.csv' %level
        record2 = pd.DataFrame(record2)
        record2.to_csv(saveName,index= False,float_format='%.2f')



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
    print("step4B end")



if __name__=="__main__":
    main()
