########################重新优化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib.animation import FFMpegWriter
import string
from copy import deepcopy
import warnings 
warnings.filterwarnings('ignore')

###################################################################################################
#
def genSamples(vehInOneLane,redVehs,speedFlagDict):
    
    samplesAll = []#收集当前车道内对应红车所有样本


    for iRed, redID in enumerate(redVehs.vehicle_id.unique()):
        #print(iRed,redID)
        
        redVehFocusTmp = redVehs[redVehs.vehicle_id == redID]#红灯状态的车辆ID
        timeList = redVehFocusTmp.timestep_time.values  

        vehInOneLane =vehInOneLane.sort_values(by='timestep_time',ascending=True)#提取持续的时间段
        maxLanePos =  max(vehInOneLane.vehicle_pos)#车道的长度
        
        
        
        for t in timeList:#枚举红灯状态下的每个时间的每一辆车 
            
      
            locTmp1 =  vehInOneLane.timestep_time == t      
            locTmp2 = abs(maxLanePos - vehInOneLane.vehicle_pos)<100 #距离红灯100米以内
            vehsAtTime = vehInOneLane[locTmp1 & locTmp2] #符合车道，时间和距离限制
                
                
            if len(vehsAtTime.vehicle_id.unique()) == 1:  # 如果当前时间车辆只有一部车，统计忽略
                #print("time",t,"len(vehsAtTime.vehicle_id.unique()) == 1")
                continue

           
            vehsAtTime =vehsAtTime.sort_values(by='vehicle_pos',ascending=False)
            counter = 0

            # 枚举当前道路上红灯状态下的每个时间的每一辆车，并生成每个时刻样本
            recordPerVeh = [];#记录每一辆车的[位置和速度]

            for rowindex, veh in vehsAtTime.iterrows():#每一辆车                  
                vehX =  veh.vehicle_x
                vehY =  veh.vehicle_y
                vehVel = veh.vehicle_speed
                vehTime  = veh.timestep_time
                vehID = veh.vehicle_id
                vehicle_Red_distane = maxLanePos - veh.vehicle_pos
                #print("vehID:",vehID," vehicle_Red_distane:",vehicle_Red_distane)

                if counter == 0 and vehID != redID:
                    #第一部车不是红灯静止车，有错误
                    print("counter == 0 and vehID != redID:",vehID,redID)
                    break

                recordPerVeh.extend([vehicle_Red_distane, vehVel])#记录每一辆车的[位置和速度]

                if counter > 0 and counter <19:#生成用于机器学习的样本,车辆不超过20
                    avg_speed_lane = 80/3.6
                    max_speed_lane = 80/3.6
                    
                    redTime = max(timeList) - t
                    realVehNum = counter
                    arrivalTimeDivRedTime = vehicle_Red_distane/max_speed_lane/(redTime+0.001)

                    #subject为主车样本
                    
                    subject = [vehID,max(timeList) - t,#主车名，红灯剩余时间
                    vehicle_Red_distane,
                    vehVel,
                    avg_speed_lane,
                    vehicle_Red_distane/(vehVel+0.01),
                    vehicle_Red_distane/avg_speed_lane,realVehNum,arrivalTimeDivRedTime]


                    #samplesTmp1为当前时刻主车前面的车（最大20车）车辆的状态
                    samplesTmp1 = deepcopy(recordPerVeh)
                    
                   
                    
                    samplesTmp1.extend([0, 0]*(19-counter))#主车前面的车（最大20车）车辆的状态


                    #samplesTmp2为当前时刻当前样本：主车+主车前面的车（最大20车）+speedFlag
                    samplesTmp2 = deepcopy(subject)#当前样本：主车
                    samplesTmp2.extend(samplesTmp1)#当前样本：主车+主车前面的车（最大20车）
                    samplesTmp2.extend([speedFlagDict[vehID]])


                    samplesAll.append(samplesTmp2)#收集所有的样本
                    
                counter = counter+1


            
            
   
    return samplesAll

###################################################################################################
#根据交通灯边红色静止车的特征，获得符合特征的红灯静止车

def extractRedVehs(vehInOneLane):
    
    redVehs = pd.DataFrame(columns=vehInOneLane.columns)  #建立空的二维数组，并且列与数据库一致。
    
    maxLanePos =  max(vehInOneLane.vehicle_pos)#车道的长度
    
    ####提取红灯车辆以及时刻
    vehInOneLane =vehInOneLane.sort_values(by='timestep_time',ascending=True)#提取持续的时间段
    timeList = vehInOneLane.timestep_time.unique()

    lowSpeedFlag = 0
    for t in timeList:#枚举每个时间，进行分析，获得红灯车辆
        
        vehsAtTime = vehInOneLane[vehInOneLane.timestep_time == float(t)]#当前时刻，当前车道上的所有车

        for index, veh in vehsAtTime.iterrows():#每一辆车
            #print(index,veh)
            vehX =  veh.vehicle_x
            vehY =  veh.vehicle_y
            vehVel = veh.vehicle_speed
            vehTime  = veh.timestep_time
            vehID = veh.vehicle_id
            vehicle_pos = veh.vehicle_pos
            if (vehVel <2/3.6) and (abs(vehicle_pos - maxLanePos)<3):#红色静止车辨别标准
                lowSpeedFlag = 1
                redVehs.loc[len(redVehs.index)] =veh
                
    return redVehs
    
###################################################################################################
#根据交通灯边红色静止车的特征，获得符合特征的所有其他车的最小速度
#认为其他车的最小速度是从计时开始到离开当前车道

def analyzingRedVehAtCurLane(redVehs,vehInOneLane,curLaneID,vehInOneEdge):
    
    speedFlagDict = dict()
    maxLanePos =  max(vehInOneLane.vehicle_pos)#车道的长度
    
    

    
    for iRed, redID in enumerate(redVehs.vehicle_id.unique()):

        redVehFocusTmp = redVehs[redVehs.vehicle_id == redID]#红灯状态的车辆ID
        timeList = redVehFocusTmp.timestep_time.values  #红灯车的持续时间

        locTmp1 = vehInOneLane.timestep_time >= min(timeList)
        locTmp2 = vehInOneLane.timestep_time <= max(timeList)
        locTmp3 = abs(maxLanePos - vehInOneLane.vehicle_pos)<100 ##距离红灯100米以内,红灯时间内车道内所有的车

        #红灯时间内车道内所有的车,而且必须距离红灯100米以内，排除绿灯时在道路的车,核心数据1
        vehsAtTimeAndDist = vehInOneLane[locTmp1 & locTmp2]
        vehIDsAtTimeAndDist = vehsAtTimeAndDist.vehicle_id.unique()#符合条件的所有车
        #print(vehsAtTimeAndDist.head(5))
        
        for ii,idTmp  in enumerate(vehIDsAtTimeAndDist):

            #提取符合ID的车，注意采用的是vehInOneLane，不是vehsAtTimeAndDist.重要！！！
            vehTmp =  vehInOneLane[vehInOneLane.vehicle_id==idTmp]
            #提取符合时间的车，  locTmp1 = vehInOneLane.timestep_time >= min(timeList)
            vehTmp = vehTmp[vehTmp.timestep_time >= min(timeList)]
            #提取符合位置的的车
            locTmp3 = abs(maxLanePos - vehTmp.vehicle_pos)<100 #100米很重要，因为有可能100开外的车才启动,速度很低
            vehTmp = vehTmp[locTmp3]
            
            if vehTmp.empty == True:
                speedFlag = -1
                speedFlagDict[idTmp] = speedFlag
                continue
                
            else:
                minSpeed = min(vehTmp.vehicle_speed.values)#距离红灯100米以内,红灯时间内一部车的是所有速度信息
                if minSpeed >= 35/3.6:
                    speedFlag  = 4
                if minSpeed <=35/3.6 and minSpeed> 25/3.6:
                    speedFlag  = 3
                if minSpeed <=25/3.6 and minSpeed> 15/3.6:
                    speedFlag  = 2
                if minSpeed <=15/3.6 and minSpeed> 5/3.6:
                    speedFlag  = 1
                if minSpeed <=5/3.6:
                    speedFlag  = 0
                #注意时间分区
                speedFlagDict[idTmp] = speedFlag
            
            ############################################
            ##vehInOneEdge再来一次提取速度标记
            ##提取符合ID的车，注意采用的是vehInOneEdge,时间规则
            vehTmp2 =  vehInOneEdge[vehInOneEdge.vehicle_id==idTmp]
            vehTmp2 =  vehTmp2[vehTmp2.timestep_time >= min(timeList)]
            
            
            ##位置规则与vehInOneLane不一样
            #我认为汽车最后时刻的距离交通灯距离大于10(也就是大于1个车长+变道最小安全距离+1秒速度值），然后不见得原因是变道
            #注意是vehInOneLane的vehTmp,不是vehInOneEdge的vehTmp2
            dist= maxLanePos-vehTmp.iloc[-1].vehicle_pos
            vel = vehTmp.iloc[-1].vehicle_speed
            
            ###对于变道情况，下面的进行了简化，非常重要
            if (dist) >(5+2+vel):#我认为汽车最后时刻的距离交通灯距离大于10(也就是大于1个车长+变道最小安全距离+1秒速度值），然后本车道上突然不见的原因是：变道
                edgeAddMaxTime = round(dist/(vel+0.001)+vel/3) #最大时间的附加时间的简易算法为距离除以速度+速度除以最大刹车速速（经验值3），因为不见的这段时间的车辆状态，难以预测
                edgeAddMaxTime = min(edgeAddMaxTime,10)#限制最大时间的附加时间为10
                vehTmp2 =  vehTmp2[vehTmp2.timestep_time <= max(timeList)+ edgeAddMaxTime]

                if vehTmp2.empty == True:
                    speedFlag1 = -1
                else:
                    minSpeed = min(vehTmp2.vehicle_speed.values)#距离红灯100米以内,红灯时间内一部车的是所有速度信息
                    if minSpeed >= 35/3.6:
                        speedFlag1  = 4
                    if minSpeed <=35/3.6 and minSpeed> 25/3.6:
                        speedFlag1  = 3
                    if minSpeed <=25/3.6 and minSpeed> 15/3.6:
                        speedFlag1  = 2
                    if minSpeed <=15/3.6 and minSpeed> 5/3.6:
                        speedFlag1  = 1
                    if minSpeed <=5/3.6:
                        speedFlag1  = 0 
                
                speedFlagDict[idTmp] = min(speedFlag,speedFlag1)
                
                
            
           
            


    return speedFlagDict

###################################################################################################
###################################################################################################
###################################################################################################
#############主程序

print("主程序：提取法国数据库的主程序。")
print("1.包括生成样本。2.计算每个样本的提取最小速度。")
print("3.保存为csv文件:france_0_allSamples1.csv")
df = pd.read_csv('./trainData/originFranceData1.csv',sep = ';')

laneList = df.vehicle_lane.unique()#获得每一条道路
numLanes = len(df.vehicle_lane.unique())


########################################################################################
#######枚举每一个车道,获得红灯附近的车，以及车道的长度
for ilane,curLaneID in enumerate(df.vehicle_lane.unique()):#枚举每一个车道
  
    print("laneIndex is %d,nameID is %s" %(ilane,curLaneID))
    
     
    if ilane<208 :
        continue
    if isinstance(curLaneID, str)  == False:
        continue
 
    
    curLaneID= laneList[ilane]

    redVehs = pd.DataFrame(columns=df.columns)  #建立空的二维数组，并且列与数据库一致。
    
    vehInOneLane = df[df.vehicle_lane==curLaneID]#获得当前车道上所有车辆
    vehInOneLane =vehInOneLane.sort_values(by='timestep_time',ascending=True)#提取持续的时间段
    
    
    #提取edge的名字,先查分割符号查#,再查_
    laneStr = curLaneID
    t1=laneStr.partition("#")
    t2=laneStr.partition("_")
   
    if t1 == laneStr:
        if t2 == laneStr:
            edgeStr=laneStr 
        else:
            edgeStr=t2[0]
    else:
        edgeStr=t1[0]
              
    resault = df['vehicle_lane'].str.contains(edgeStr)
    resault.fillna(value=False,inplace = True)
    vehInOneEdge = df[resault]#获得当前edge上所有车辆
    vehInOneEdge = vehInOneEdge.sort_values(by='timestep_time',ascending=True)#提取持续的时间段
                      
    
    if vehInOneLane.empty == True:#当前车道没有车
        continue
        
    ###提取交通灯边的静止车    
    redVehs = extractRedVehs(vehInOneLane) 
    if redVehs.empty == True:#如果当前车道没有红灯车
         continue

            
    ####给出每辆车的最小速度        
    speedFlagDict = analyzingRedVehAtCurLane(redVehs,vehInOneLane,curLaneID,vehInOneEdge)
    

    print("#生成样本")
    samplesAll=genSamples(vehInOneLane,redVehs,speedFlagDict) 
        
    #print("len(speedFlagDict):",len(speedFlagDict1))
    print("len(redVehs):",len(redVehs))
    print("len(samplesAll):",len(samplesAll))
        
    #当前车道，每个红灯车的所有时刻的样本
    name1 = ["vehID","redLightTime","distToRedLight","speed","laneAvgSpeed","arriveTime1","arriveTime2","numStillVeh","ArrTimeDivRedTime"]   
    name2 = ["vehPos_1","vehSpeed_1","vehPos_2","vehSpeed_2","vehPos_3","vehSpeed_3","vehPos_4","vehSpeed_4"] 
    name3 = ["vehPos_5","vehSpeed_5","vehPos_6","vehSpeed_6","vehPos_7","vehSpeed_7","vehPos_8","vehSpeed_8"]
    name4 = ["vehPos_9","vehSpeed_9","vehPos_10","vehSpeed_10","vehPos_11","vehSpeed_11","vehPos_12","vehSpeed_12"]
    name5 = ["vehPos_13","vehSpeed_13","vehPos_14","vehSpeed_14","vehPos_15","vehSpeed_15","vehPos_16","vehSpeed_16"]
    name6 = ["vehPos_17","vehSpeed_17","vehPos_18","vehSpeed_18","vehPos_19","vehSpeed_19","vehPos_20","vehSpeed_20"]
    
    headers = name1+name2+name3+name4+name5+name6+["speedFlag"]

    if samplesAll != []:
        #print(samplesAll[0])
        samplesTmp = pd.DataFrame(samplesAll,columns=headers)
        print(samplesTmp.info())
        filename = './franceRedData/'+str(ilane)+'+'+curLaneID+'.csv'
        samplesTmp.to_csv(filename,float_format='%.3f',index=0) 


###################################################################################
#将所有样本集合成一个CSV文件
import os
import pandas as pd
path = "./franceRedData/"

filelist = [path + i for i in os.listdir(path)]
dataset = pd.read_csv(filelist[0])

for tmpFile in filelist:
    if tmpFile.endswith(".csv"):
        #print(tmpFile)
        tmpDF = pd.read_csv(tmpFile)
        dataset = pd.concat([dataset,tmpDF],ignore_index=True,axis=0)
        

filename= "./trainData/"+"0_allSamples.csv" 
dataset.to_csv(filename,float_format='%.3f',index=0) 
        

