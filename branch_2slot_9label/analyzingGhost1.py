
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

def analyzX(tmp):
    
    len1 = len(tmp)
    redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2,laneID,ArrivalDivRedTime,\
        vehPos_1,vehSpeed_1,vehPos_2,vehSpeed_2,vehPos_3,vehSpeed_3,vehPos_4,vehSpeed_4,vehPos_5,vehSpeed_5,\
        vehPos_6,vehSpeed_6,vehPos_7,vehSpeed_7,vehPos_8,vehSpeed_8,vehPos_9,vehSpeed_9,vehPos_10,vehSpeed_10,\
        vehPos_11,vehSpeed_11,vehPos_12,vehSpeed_12,vehPos_13,vehSpeed_13,vehPos_14,vehSpeed_14,vehPos_15,vehSpeed_15,\
        vehPos_16,vehSpeed_16,vehPos_17,vehSpeed_17,vehPos_18,vehSpeed_18,vehPos_19,vehSpeed_19,vehPos_20,vehSpeed_20 = tmp
    vehObj = np.array([distToRedLight,speed])
    vehsOthers = tmp[8:len1]
    
    vehsOthers = vehsOthers.reshape(-1,2)
    vehsOthers_all = vehsOthers[np.where(vehsOthers[:,0]>0)]
    
    vehsOthers1 = vehsOthers_all[0:-1]#最后一个是目标车，不要
    #print("vehObj:",np.round(vehObj,3))
    #print("vehs_all",np.round(vehsOthers_all,2))#数据命名错误，vehsOthers等于车道上所有车
    #print("vehsOthers1",np.round(vehsOthers1,2))
    
    return vehObj,vehsOthers1,redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2,laneID,ArrivalDivRedTime
    
    
    
dfSumoData = pd.read_csv('sumoSimData.csv', sep=',')

headSumoData = ['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                               'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                               'smv1','smv2']

dfSimVehParams = pd.read_csv('paramsVehAll.csv', sep=',')


headSimVehParams = ['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                       'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                               'minSpeed0']
fpk=open('lowprobSamples.pkf','rb')   
[xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
fpk.close()  

nSamples,nFeatures = dfSumoData.shape
print("dfSumoData.shape:",dfSumoData.shape)
nSamples,nFeatures = dfSimVehParams.shape
print("dfSimVehParams.shape:",dfSimVehParams.shape)


nSamples = 2000
analy1 = np.array([0.0,0.0,0.0,0.0])
numAnaly1 = 0

analy2 = np.array([0.0,0.0,0.0,0.0,0.0])
numAnaly2 = 0

analy2B = np.array([0.0])

 
for j in range(nSamples):
    sampleIndex =  dfSumoData.iloc[j]['sampleIndex'].item()
    sampleIndex = int(sampleIndex)
    originOutput =  dfSumoData.iloc[j]['originOutput'].item()
    sumoOutputSpeedTag =  dfSumoData.iloc[j]['sumoOutputSpeedTag'].item()

    print('j########################################################')
    print('j: ',j)
    print('originOutput:',originOutput)
    print('sumoOutputSpeedTag:',sumoOutputSpeedTag)
    print('sampleIndex:',sampleIndex)
      
     #论文用
     ##预测不停车过路口停车线
     #第一种情况，直接给出结论，不能不停车过路口并给出SUMO仿真验证
    if 1 and originOutput == 0 and sumoOutputSpeedTag== 0 :
            
            print("originOutput == 0 and sumoOutputSpeedTag== 0")
            print("第一种情况，直接给出结论，不能不停车过路口并给出SUMO仿真验证")
            indT= dfSimVehParams['sampleIndex'] == sampleIndex
            dfSVP = dfSimVehParams[indT]
            print('dfSVP.shape:',dfSVP.shape)
            
            tmp1 = (dfSVP['maxSpeed0']>20/3.6) & (dfSVP['maxSpeed']>20/3.6)  & (dfSVP['reacTime']<0.8)
            dfSVP1 = dfSVP[tmp1]
            dfSVP1 = dfSVP1[['maxSpeed0','minSpeed0']]
            print('dfSVP1.shape:',dfSVP1.shape)
            numAnaly1 =  numAnaly1 +dfSVP1.shape[0]
            
            tmp1 =  dfSVP1[dfSVP1['minSpeed0']<5/3.6]
            analy1[0]= analy1[0]+tmp1.shape[0]
            print("minSpeed0<5/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
            tmp  = (dfSVP1['minSpeed0']<10/3.6) & (dfSVP1['minSpeed0']>=5/3.6)
            tmp1 =  dfSVP1[tmp]
            if tmp1.empty == False:
                analy1[1]= analy1[1]+tmp1.shape[0]
                print("5/3.6<=minSpeed0<10/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
            tmp  = (dfSVP1['minSpeed0']<15/3.6) & (dfSVP1['minSpeed0']>=10/3.6)
            tmp1 =  dfSVP1[tmp]
            if tmp1.empty == False:
                analy1[2]= analy1[2]+tmp1.shape[0]
                print("10/3.6<=minSpeed0<15/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
                
            tmp  = (dfSVP1['minSpeed0']<20/3.6) & (dfSVP1['minSpeed0']>=15/3.6)
            tmp1 =  dfSVP1[tmp]    
            if tmp1.empty == False: 
                analy1[3]= analy1[3]+tmp1.shape[0]
                print("15/3.6<=minSpeed0minSpeed0<20/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
                
     #论文用
     ##预测不停车过路口停车线
     #第二种情况，根据仿真结果，给出不同速度下能否不停车通过路口停止线,如何验证

    if 0 and originOutput == 0 and sumoOutputSpeedTag >0:
            
            print("originOutput == 0 and sumoOutputSpeedTag > 0")
            print("第二种情况，根据仿真结果，给出不同速度下能否不停车通过路口停止线")
            indT= dfSimVehParams['sampleIndex'] == sampleIndex
            dfSVP = dfSimVehParams[indT]
            print('dfSVP.shape:',dfSVP.shape)
            
            tmp1 = (dfSVP['reacTime0']<0.8) & (dfSVP['reacTime']<0.8)
            dfSVP1 = dfSVP[tmp1]
            dfSVP1 = dfSVP1[['maxSpeed0','minSpeed0']]
            print('dfSVP1.shape:',dfSVP1.shape)
            
            numAnaly2 =  numAnaly2 +dfSVP1.shape[0]#符合条件的例子总数
            
            tmp  = (dfSVP1['maxSpeed0']<20/3.6) & (dfSVP1['minSpeed0']>=15/3.6)
            tmp1 =  dfSVP1[tmp]
            analy2[0]= analy2[0]+tmp1.shape[0]
            print("maxSpeed0<20/3.6 minSpeed >15/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
            tmp  =(dfSVP1['maxSpeed0']>20/3.6) & (dfSVP1['maxSpeed0']<30/3.6) & (dfSVP1['minSpeed0']>=15/3.6)
            tmp1 =  dfSVP1[tmp]
            if tmp1.empty == False:
                analy2[1]= analy2[1]+tmp1.shape[0]
                print("maxSpeed0<30/3.6 minSpeed >15/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
            tmp  = (dfSVP1['maxSpeed0']>30/3.6) & (dfSVP1['maxSpeed0']<40/3.6) & (dfSVP1['minSpeed0']>=15/3.6)
            tmp1 =  dfSVP1[tmp]
            if tmp1.empty == False:
                analy2[2]= analy2[2]+tmp1.shape[0]
                print("maxSpeed0<40/3.6 minSpeed >15/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
                
            tmp  = (dfSVP1['maxSpeed0']>40/3.6) & (dfSVP1['maxSpeed0']<50/3.6) & (dfSVP1['minSpeed0']>=15/3.6)
            tmp1 =  dfSVP1[tmp]    
            if tmp1.empty == False: 
                analy2[3]= analy2[3]+tmp1.shape[0]
                print("maxSpeed0<50/3.6 minSpeed >15/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
            tmp  = (dfSVP1['maxSpeed0']>50/3.6) & (dfSVP1['maxSpeed0']<60/3.6) & (dfSVP1['minSpeed0']>=15/3.6)
            tmp1 =  dfSVP1[tmp]    
            if tmp1.empty == False: 
                analy2[4]= analy2[4]+tmp1.shape[0]
                print("maxSpeed0<60/3.6 minSpeed >15/3.6: ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
            
            tmp  = dfSVP1['minSpeed0']<5/3.6
            tmp1 =  dfSVP1[tmp]    
            if tmp1.empty == False: 
                analy2B[0]= analy2B[0]+tmp1.shape[0]#符合条件的例子总数
                print("'minSpeed0': ",tmp1.shape[0]/dfSVP1.shape[0]*100)
            
  
            
             
     ##论文用
     ##幽灵停车
    if 0 and originOutput == 0 and sumoOutputSpeedTag >0:
            print("#########################################")
            print("幽灵停车分析")
            indT= dfSimVehParams['sampleIndex'] == sampleIndex
            dfSVP = dfSimVehParams[indT]
            print('dfSVP.shape:',dfSVP.shape)
           
            
            tmp = xlowpra[sampleIndex][0:48]
            vehObj,vehsOthers1,redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2,laneID,ArrivalDivRedTime = analyzX(tmp)
            print('vehsOthers1.shape: ',vehsOthers1.shape)
            print('vehObj: ',vehObj)
            print('vehsOthers1 ',vehsOthers1)
            print('redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2\n')
            print(redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2)
         
            '''如果实际为0，SUMO大于0，有排队车辆，同时按照车道平均速度到达停车线的时间在小于红灯时间，就是红灯太长了，导致实际为0")'''
            '''SUMO大于0，是因为目标车辆的设置最大速度有一定的压制")'''
            
            nVehsFront = vehsOthers1.shape[0]
            if nVehsFront>0 and arriveTime2>redLightTime:#按照车道平均速度到达停车线的时间在大于红灯时间,有排队车辆，
                print("如果实际为0，SUMO大于0，有排队车辆，同时按照车道平均速度到达停车线的时间在大于红灯时间，可以理解为幽灵停车")
                tmp1 =   (dfSVP['reacTime']>1.1)
                dfSVP1 = dfSVP[tmp1]
                dfSVP1 = dfSVP1[['maxSpeed0','minSpeed0']]
                print('dfSVP1.shape:',dfSVP1.shape)
                print('nVehsFront:',nVehsFront)
                print('dfSVP1[reacTime]>1:\n',dfSVP1)
                print(dfSVP1.mean())
                
                plt.scatter(dfSVP1['maxSpeed0'], dfSVP1['minSpeed0'],marker='o',label='>1.1')
                plt.title('maxSpeed0 and minSpeed0')
                plt.xlabel('maxSpeed0')
                plt.ylabel('minSpeed0')
                plt.savefig('tmp1.png')  # 保存为png格式图片
                
               
                
                #fs = "tmp1ForPlot1.csv"
                #dfSVP1.to_csv(fs,index= False, header= False)
                
                tmp1 =   (dfSVP['reacTime']<0.3)
                dfSVP1 = dfSVP[tmp1]
                dfSVP1 = dfSVP1[['maxSpeed0','minSpeed0']]
                print('dfSVP1.shape:',dfSVP1.shape)
                print('nVehsFront:',nVehsFront)
                print('dfSVP1[reacTime]<0.3:\n',dfSVP1)
                print(dfSVP1.mean())
                
                plt.scatter(dfSVP1['maxSpeed0'], dfSVP1['minSpeed0'],marker='+',label='<0.3')
                plt.title('maxSpeed0 and minSpeed0')
                plt.xlabel('maxSpeed0')
                plt.ylabel('minSpeed0')
                plt.legend()
                plt.savefig('tmp2.png')  # 保存为png格式图片
               
                plt.close()
                input()
                
                
    #幽灵停车对比           
    if 0 and originOutput > 0 and sumoOutputSpeedTag >0:
            print("#########################################")
            print("幽灵停车对比分析")
            indT= dfSimVehParams['sampleIndex'] == sampleIndex
            dfSVP = dfSimVehParams[indT]
            print('dfSVP.shape:',dfSVP.shape)
           
            
            tmp = xlowpra[sampleIndex][0:48]
            vehObj,vehsOthers1,redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2,laneID,ArrivalDivRedTime = analyzX(tmp)
            print('vehsOthers1.shape: ',vehsOthers1.shape)
            print('vehObj: ',vehObj)
            print('vehsOthers1 ',vehsOthers1)
            print('redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2\n')
            print(redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2)
         
           
            
            nVehsFront = vehsOthers1.shape[0]
            if nVehsFront>0 and arriveTime2>redLightTime:#按照车道平均速度到达停车线的时间在大于红灯时间,有排队车辆，
                print("如果实际为0，SUMO大于0，有排队车辆，同时按照车道平均速度到达停车线的时间在大于红灯时间，可以理解为幽灵停车")
                tmp1 =   dfSVP['reacTime']>1.1
                dfSVP1 = dfSVP[tmp1]
                dfSVP1 = dfSVP1[['maxSpeed0','minSpeed0']]
                print('dfSVP1.shape:',dfSVP1.shape)
                print('nVehsFront:',nVehsFront)
                print('dfSVP1[reacTime]>1:\n',dfSVP1)
                print(dfSVP1.mean())
                
                plt.scatter(dfSVP1['maxSpeed0'], dfSVP1['minSpeed0'],marker='o',label='>1.1')
                plt.title('maxSpeed0 and minSpeed0')
                plt.xlabel('maxSpeed0')
                plt.ylabel('minSpeed0')
                plt.savefig('tmp3.png')  # 保存为png格式图片
                
                tmp1 =   dfSVP['reacTime']<0.3
                dfSVP1 = dfSVP[tmp1]
                dfSVP1 = dfSVP1[['maxSpeed0','minSpeed0']]
                print('dfSVP1.shape:',dfSVP1.shape)
                print('nVehsFront:',nVehsFront)
                print('dfSVP1[reacTime]<0.3:\n',dfSVP1)
                print(dfSVP1.mean())
                
                plt.scatter(dfSVP1['maxSpeed0'], dfSVP1['minSpeed0'],marker='+',label='>1.1')
                plt.title('maxSpeed0 and minSpeed0')
                plt.xlabel('maxSpeed0')
                plt.ylabel('minSpeed0')
                plt.savefig('tmp4.png')  # 保存为png格式图片
                plt.close()
                input()
'''  
print('###################################################')
print(analy1)
print(numAnaly1)
print("第一种情况，不能不停车过路口的比例：",np.round(analy1*100/(numAnaly1+0.00001),3))

print('###################################################')
print(analy2)
print(numAnaly2)
print("第二种情况，各种最大速度下的比例：",np.round(analy2*100/(numAnaly2+0.00001),3))

print('###################################################')
print(analy2B)
print("第二种情况，小于5/3.6的比例：",np.round(analy2B*100/(numAnaly2+0.00001),3))
'''