#####################################################################################
#####################################################################################
#####################################################################################
#用于简单过路灯模拟
#1.单直车道. 2.无车道转换。3.所有车假设为同一类车，也就是汽车动力学和汽车运动学一样
#作者lukeliuli@163.com
# -*- coding: utf-8 -*-
#https://sumo.dlr.de/pydoc/traci.html
#@author: lukeliuli@163.com

print("#########################################################################")
print("step2:基于SUMO和蒙特卡洛模拟分析生成预测当前样本的最小速度，原始对应程序为mainSimSumoFranceData，需要在docker的原始环境下运行，没有conda")
print("程序标号3，以及是并行仿真版本")


#用于简单过路灯模拟
#1.单直车道. 2.无车道转换。3.所有车假设为同一类车，也就是汽车动力学和汽车运动学一样
#作者lukeliuli@163.com
# -*- coding: utf-8 -*-
#https://sumo.dlr.de/pydoc/traci.html
"""
@author: lukeliuli@163.com
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import sys
import random

#import traci as libsumo
import libsumo
import matplotlib.pyplot as plt  


import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
import numpy as np
import os
import time
import argparse  
########################################################################################################################
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



def simSumoCmd(params):
    #print(params)
    simNum =  params["simNum"] #= 150
    redLightTime =  params["redLightTime"]# = 15
    otherVehs =   params["otherVehs"] #= [[1, 0],[7,0],[14,0]]  # [距离交通灯的距离1，行驶速度1,距离交通灯的距离2，行驶速度2]
    otherVehsParams =   params["otherVehsParams"] #= [5.5, 2, -9, 60 /3.6, 0.2]  # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距]
    objectVeh =  params["objectVeh"] #= [50, 0] 距离交通灯的距离,当前速度
    objectVehParams =  params["objectVehParams"] #= [5.5, 2, -9, 60 /3.6, 0.2]  # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距]
    
    vehsInitNums = 1+len(otherVehs)
    
    nextTLSID = "2"
    trafficLightPos =200
    objvehID  = 'o1'
    routeID = 'platoon_route'
    typeID1 = 'EBUS'
    typeID2 = 'EBUS2'
    otherVehID = 'd'


    #####基本运行配置
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    
    path = os.getcwd()
    path ='/home/liuli/myCodes/SUMONBDT/'
    #path2 = path+"\sumoCfgs\my1Lane1Tls-server.sumocfg"#window
    path2 = path+"/sumoCfgs1/my1Lane1Tls-server.sumocfg"#linux
    sumoBinary = "sumo"
    
    libsumo.close()
    #libsumo.start(["sumo-gui", "-c", path2])
    libsumo.start([sumoBinary, "-c", path2,'--no-warnings'])
    #libsumo.start([sumoBinary, "-c", path2])
    #####基本车辆参数设定
    
        
     # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                               
    length,maxAcc,maxDacc,maxSpeed0,tau,minGap,imperfection,speedFactor = objectVehParams
    
    libsumo.vehicletype.setAccel(typeID1,maxAcc)  
    libsumo.vehicletype.setDecel(typeID1,maxDacc) 
    libsumo.vehicletype.setImperfection(typeID1,imperfection)
    libsumo.vehicletype.setLength(typeID1,length) 
    libsumo.vehicletype.setMaxSpeed(typeID1,maxSpeed0)   
    libsumo.vehicletype.setMinGap(typeID1,minGap) 
    libsumo.vehicletype.setSpeedFactor(typeID1,speedFactor)  
    libsumo.vehicletype.setTau(typeID1,tau)  
  
                          
                            
    # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,速度噪声]                         
    length,maxAcc,maxDacc,maxSpeed1,tau,minGap,imperfection,speedFactor = otherVehsParams
    libsumo.vehicletype.setAccel(typeID2,maxAcc)  
    libsumo.vehicletype.setDecel(typeID2,maxDacc) 
    libsumo.vehicletype.setImperfection(typeID2,imperfection)
    libsumo.vehicletype.setLength(typeID2,length) 
    libsumo.vehicletype.setMaxSpeed(typeID2,maxSpeed1)   
    libsumo.vehicletype.setMinGap(typeID2,minGap) 
    libsumo.vehicletype.setSpeedFactor(typeID2,speedFactor) 
    libsumo.vehicletype.setTau(typeID2,tau) 
                 
                            


                                 

    #greenTime = max(random.random()*greenTime,0.1)
    #交通灯设定为红灯逻辑
    allProgramLogics = libsumo.trafficlight.getAllProgramLogics(nextTLSID)
    lgc1 = allProgramLogics[0]
    yellowDurTime = lgc1.phases[0].duration
    greenDurTime = lgc1.phases[1].duration
    yellowDurTime = lgc1.phases[2].duration
    redDurTime = lgc1.phases[3].duration
    
    lgc1.phases[0].state = 'r'
    lgc1.phases[0].duration =yellowDurTime
    lgc1.phases[1].state = 'r'
    lgc1.phases[1].duration =redLightTime#修改绿灯时间
    
    lgc1.phases[2].state = 'y'
    lgc1.phases[2].duration =yellowDurTime
    
    lgc1.phases[3].state = 'G'
    lgc1.phases[3].duration =greenDurTime
    
    libsumo.trafficlight.setProgramLogic(nextTLSID,lgc1) 
    phaseDur = libsumo.trafficlight.getPhaseDuration(nextTLSID)
    
    """
    ####修改逻辑和红灯时间，随机参数
    redTime =  params["redTime"]#注意坐标的原点
    redTime = max(random.random()*redTime,0.1)
    allProgramLogics = libsumo.trafficlight.getAllProgramLogics(nextTLSID)
    lgc1 = allProgramLogics[0]
    yellowDurTime = lgc1.phases[0].duration

    lgc1.phases[1].state = 'r'
    redDurTime = lgc1.phases[1].duration

    yellowDurTime = lgc1.phases[2].duration

    lgc1.phases[3].state = 'G'
    greenDurTime = lgc1.phases[3].duration

    lgc1.phases[1].duration = redTime
    libsumo.trafficlight.setProgramLogic(nextTLSID,lgc1) 
    phaseDur = libsumo.trafficlight.getPhaseDuration(nextTLSID)
    
    
    veh = "00003"
    tlsList = libsumo.vehicle.getNextTLS(veh)
    tlsID_T, tlsIndex, dist, state = tlsList[0] 
    """
    requireStop =0 
    stepNum = 0
    libsumo.simulationStep()
    statRec1 = []
    minSpeed = 500/3.6
    ##############################################################
    while requireStop == 0:
        
        if (stepNum == 0):
            dist,vel = objectVeh
            vel  = min([40/3.6,vel,maxSpeed0])
            libsumo.vehicle.add(objvehID, routeID, typeID=typeID1, depart='0', departLane='first', \
                        departPos=str(trafficLightPos-dist), departSpeed=str(vel))
            counter = 0
           
            for v in otherVehs:
                counter = counter+1
                dist,vel = v
                vel  = min([40/3.6,vel,maxSpeed1])
                #https://sumo.dlr.de/docs/Specification/index.html
                #https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html
                libsumo.vehicle.add(otherVehID+str(counter), routeID, typeID=typeID2, depart='0', departLane='first', \
                                    departPos=str(trafficLightPos-dist), departSpeed=str(vel))
                
        
        libsumo.simulationStep()
        stepNum += 1
        
        if stepNum==1 and libsumo.vehicle.getIDCount() != vehsInitNums:#因为各种原因无法在模拟中插入车辆，例如跟车模型限制
            print("起始模拟车辆数目错误，无法在模拟中插入足够车辆")
            requireStop = -1
            statRec1 = []
            strRec1 = []
            minSpeed = -1
            leaderInfo = []
             
            break
            
        if stepNum==1:
            vehs = libsumo.vehicle.getIDList()
            for i in range(libsumo.vehicle.getIDCount()):
                libsumo.vehicle.setSpeedFactor(vehs[i],1)

        ####https://sumo.dlr.de/daily/pydoc/traci._vehicle.html
        states = libsumo.trafficlight.getRedYellowGreenState(nextTLSID)
        timeT = libsumo.simulation.getCurrentTime()  # 当前时间
        nextSwitch = libsumo.trafficlight.getNextSwitch(nextTLSID)
        phaseName = libsumo.trafficlight.getPhaseName(nextTLSID)
        phase = libsumo.trafficlight.getPhase(nextTLSID)
        phaseDur = libsumo.trafficlight.getPhaseDuration(nextTLSID)
        curTime = timeT/1000
        phaseLeftTime = nextSwitch - curTime

        nextTLSNow= libsumo.vehicle.getNextTLS(objvehID)

        vehPos = libsumo.vehicle.getPosition(objvehID)[0]
        speed = libsumo.vehicle.getSpeed(objvehID)
        vehMaxSpeed = libsumo.vehicle.getMaxSpeed(objvehID)
        vehLanePos = libsumo.vehicle.getLanePosition(objvehID)

        laneID = libsumo.vehicle.getLaneID(objvehID)
        
        
        
        edgeID = libsumo.vehicle.getRoadID(objvehID)
        #meanSpeed = libsumo.lane.getLastStepMeanSpeed(laneID)
        #laneMaxSpeed = libsumo.lane.getMaxSpeed(laneID)
        #vehMaxSpeed = min(vehMaxSpeed, laneMaxSpeed)

        dist2TLS = trafficLightPos-vehLanePos
        arrivalTime1 = min(100,dist / (speed + 0.001))
        arrivalTime2 = min(100,dist / vehMaxSpeed)


        passTLS = nextTLSNow==nextTLSID

        leaderID = libsumo.vehicle.getLeader(objvehID)
        if leaderID:
            leaderInfo = getVehicleInfo(leaderID[0])
        else:
            leaderInfo = None
        
        
        #if timeT> 40000:#毫秒ms
        #    print("simStop:timeT/1000 > 40")
        #    requireStop = 1
            
        if dist2TLS<=0.3: 
            #print("simStop:dist2TLS<=0.3")
            requireStop = 1
            
            
        edgeIDNow =  libsumo.vehicle.getRoadID(objvehID)
        if edgeIDNow != "e1to2" or passTLS:  # if time is over 30 second stoping the simulation
            #print("simStop:next tls,edge is %s,%s"% (nextTLSNow,libsumo.vehicle.getRoadID(objvehID)))
            requireStop = 1
            
            
      

        
        tmp = "running time(ms):%d,tls state: %s,duration:%.3f," % (timeT,states,phaseDur)
        strRec1 = tmp
        tmp = "speed:%.2f,lanePos:%s,dist2TLS:%.2f," % (speed,vehLanePos,dist2TLS)
        strRec1 = strRec1+tmp
        tmp = "edgeID is %s, TLS is %s," % (edgeIDNow,nextTLSNow)
        strRec1 = strRec1+tmp
        
        statRec1.append([states,phaseDur,timeT,passTLS,speed,vehLanePos,dist2TLS])
        
        if timeT>1000:#降低误差
            minSpeed = min(speed,minSpeed)
        
        if params["log"]:
            print(strRec1)  
            print("leadID：%s,leadInfo:%s" %(leaderID[0],leaderInfo))
            vehs = libsumo.vehicle.getIDList()
            for i in range(libsumo.vehicle.getIDCount()):
                allowMaxSpeed=libsumo.vehicle.getAllowedSpeed(vehs[i])
                print(vehs[i]," allowedMaxSpeed:",allowMaxSpeed)
                maxSpeed=libsumo.vehicle.getMaxSpeed(vehs[i])
                print(vehs[i]," maxSpeed:",maxSpeed)

            
        #print("步数:%d,已经在道路上的车辆数:%d" %(stepNum,libsumo.vehicle.getIDCount()))
            
        #print("running time(ms):%d,tls state: %s,duration:%.3f," % (timeT,states,phaseDur))
        #print("speed:%.2f,lanePos:%s,dist2TLS:%.2f," % (speed,vehLanePos,dist2TLS))

    
    libsumo.close()
                  
    return statRec1,strRec1,minSpeed,leaderInfo,requireStop 
########################################################################################################################

def getVehicleInfo(objvehID):
    info ="No Veh"
    if len(objvehID) == 0:
        return info
     
    if objvehID:
        vehPos = libsumo.vehicle.getPosition(objvehID)[0]
        speed = libsumo.vehicle.getSpeed(objvehID)
        vehMaxSpeed = libsumo.vehicle.getMaxSpeed(objvehID)
        vehLanePos = libsumo.vehicle.getLanePosition(objvehID)
        laneID = libsumo.vehicle.getLaneID(objvehID)
        edgeID = libsumo.vehicle.getRoadID(objvehID)

        info = "ID:%s,vehPos:%.2f,speed:%.2f,vehLanePos:%s,laneID:%s,edgeID:%s," %(objvehID,vehPos,speed,vehLanePos,laneID,edgeID)

        TLSNow = libsumo.vehicle.getNextTLS(objvehID)
        if len(TLSNow)>0:
            name= TLSNow[0][0]
            phase= TLSNow[0][1]
            dist= TLSNow[0][3]
            tmp = "TLSName:%s,TLSPhase:%s,TLSDist:%s" %(name,phase,dist)
            info = info+tmp


    return info


########################################################################################################################
###测试程序2,从pickle中读取样本,注意与france csv不一样，输入数据只有48，把vehID,speedFlag 已经去掉了
import pickle



def configAndRun(tmp,index,simNum =10):
    
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
    print("vehObj:",np.round(vehObj,3))
    print("vehs_all",np.round(vehsOthers_all,2))#数据命名错误，vehsOthers等于车道上所有车
    print("vehsOthers1",np.round(vehsOthers1,2))
    
    #if vehsOthers1.shape[0] == 0:
    #    return
    #print(vehsOthers)
    #redLightTime = redLightTime

    print("根据实际情况，我们认为红灯变绿灯时，红灯前停止的驾驶员的反应时间和车辆启动时间为1秒")
    print("而模拟中车辆的启动时间很快，能1秒内加速到2m/s，所以模拟中实际红灯时间加1.5秒,(暂时不改)")
    #if vehsOthers_all[0][1] <1/3.6 and vehsOthers_all[0][0] >0:#头车速度低于1km/h
    #    redLightTime= redLightTime+1.5
    #if vehsOthers_all.shape[0]>1  and vehsOthers_all[1][1] <1/3.6 and vehsOthers_all[1][0] >0:#存在第二步车，而且速度低于1km/h
    #    redLightTime= redLightTime
    #print("redLightTime",redLightTime)
    
    #time.sleep(5);
    #####################################################################
    
    params =dict()
    params["simNum"] = simNum
    
    params["otherVehs"] = vehsOthers1  # [[距离交通灯的距离1，行驶速度1],[距离交通灯的距离2，行驶速度2]]

    #[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                              
    params["otherVehsParams"] = [4,2,9,60/3.6,     0.5, 0.5 ,0.01,0.05] 

    params["objectVeh"] = vehObj
    #[车辆长度，最大加速度,最大减加速度，最大速度，反应时间(0.01到0.1的传输延迟，0.2到0.5的执行延迟),最小间距,不专心,速度噪声]  
    params["objectVehParams"] = [4,2,9,60/3.6,                       0.5,                           0.5,      0.01,  0.05] 
    params["log"] = False
    minSpeedList = []
    
    paramsVehList = []
    emptyRun = 0#因为任何应用空转
    #for i in range(params["simNum"]):
    while(len(minSpeedList)<params["simNum"])  and emptyRun < 250:
          print("\nsampleIndex: %d,simNum:%d Start" %(index,len(minSpeedList)))
          #加入噪声
          params["otherVehsParams"] = [2,random.uniform(1,2),9,random.uniform(45/3.6,70/3.6),random.uniform(0.01,0.5), 0.1 ,0.00,0.00] 
          params["objectVehParams"] = [2,random.uniform(1,2),9,random.uniform(45/3.6,70/3.6),random.uniform(0.01,0.5), 0.1 ,0.00,0.00] 



          #随机0.5秒为驾驶员的反应时间和车辆启动时间的附加随机值
          params["redLightTime"] = float(redLightTime)

          statRec1,strRec1,minSpeed,leaderInfo,requireStop  = simSumoCmd(params)
     

          #print("simNum:%d End" %i)
          #print(strRec1)
          #print("leaderInfo",leaderInfo )
          #print('minSpeed',minSpeed)
          #print('requireStop',requireStop)

          if requireStop<0:
              emptyRun =  emptyRun+1 
              continue
             
          if minSpeed >=0:
              minSpeedList.append(minSpeed)
              paramsVehListTmp = [index]
              paramsVehListTmp.extend(params["objectVehParams"])
              paramsVehListTmp.extend(params["otherVehsParams"])
              paramsVehListTmp.extend([minSpeed])
              #print('\n paramsVehListTmp',paramsVehListTmp)
              paramsVehList.append(paramsVehListTmp)

             

    minSpeedList1 = np.array(minSpeedList)
    if len(minSpeedList)>0:
        print("MonteCarloSimulation %d times,minSpeedList ,min:%.2f,max:%.2f,mean:%.2f" %(len(minSpeedList),np.min(minSpeedList1),np.max(minSpeedList1),np.mean(minSpeedList1)))

   
    #plt.hist(minSpeedList)
     
    return minSpeedList1,paramsVehList
  
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
#主程序



def test4ParallSim1(sections,labels,level,simNum,numRunSample):
    '''从test3,修改而来，主要修改地方为根据输入参数进行样本分割，并将结果进行保存到独立的文件夹中'''
    print("test4ParallSim1,从lowprobSamples.pkf，样本已经进行[x,y]分割.样本为2slot,5hier,9label\n\n")
    print("test4ParallSim1,样本section为:")
    print(sections)
    strTmp = 'step1_lowprobSamplesLevel%d.pkf' %level
    fpk=open(strTmp,'rb')   
    [xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
    fpk.close()   
    
   
    
    numSamples,numFeatures = xlowpra.shape
    
    numSamples = min(numSamples,numRunSample)#numRunSample用小样本测试程序是否可行
    
    strLog = "numSamples %d,numFeatures%d" %(numSamples,numFeatures)
    print(strLog)
   

    rvl = []
  
    paramsVehAll = []
    #simNum =  500  #sim==500,运行一个样本时间为10秒
    
 
    stt = round(numSamples*sections[0])
    end = round(numSamples*sections[1])
    jTmpIndex = np.arange(stt,end)
    
    strLog = "start:%d,end:%d" %(stt,end)
    print(strLog)
   
  
    for j in jTmpIndex:
        
        #xlowpra:x-name = 8(keyFeature)+40(otherVehcle)+7(keyFeatures)+39(otherVehs)= 94
        tmp = xlowpra[j][0:48]

        minSpeedList1,paramsVehList = configAndRun(tmp,j,simNum)

        if len(minSpeedList1) == 0:
            continue
            
        
        outputAvgSpeed = np.round(np.mean(minSpeedList1),2)
        sumoOutputSpeedTag = minSpeed2Tag(outputAvgSpeed)
       
        originOutput = ylowpraLabel[j][0]
        kerasPredictLabel = ylowPredictLabel[j][0]
        kerasPredictNN = ylowpraPredictNN[j]
        #[i = 0,outputAvgSpeed=1,outputY[0]=2,outputSpeedTag=3,ylowPredictLabel[i][0]=4]

        sumoRVL=[j,outputAvgSpeed,originOutput,sumoOutputSpeedTag,kerasPredictLabel]
        sumoRVL.extend(kerasPredictNN)
        sumoRVL.extend(np.round(minSpeedList1[0:2],2))
        
        rvl.append(sumoRVL)
        
        paramsVehAll.extend(paramsVehList)   
        
        timeNow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        strLog = "\nsampleIndex:%d, time End %s" %(j,timeNow)
        print(strLog)
       


            
            
     
     
    df = pd.DataFrame(rvl)
    fs ="./tmpData/"+labels+"sumoSimData.csv"
    #[5+2+9]
    if level == 2:
        df.to_csv(fs,index= False, header=['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                               'NN0','NN1','NN2','NN3','smv1','smv2'])
        
    if level == 7:
        df.to_csv(fs,index= False, header=['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                           'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8','smv1','smv2'])
    df = pd.DataFrame(paramsVehAll)
    fs = "./tmpData/"+labels+"paramsVehAll.csv"
    df.to_csv(fs,index= False, header=['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                       'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                               'minSpeed0'])
                                  
  

######################################################################################################################## 
########################################################################################################################   
########################################################################################################################

from multiprocessing import Process
from datetime import datetime
import math
import time

def main():
    
    #需要再root而不是GPU环境下运行
    #python3 mainSimpleStep2_sumoSimByFranceParallSim1.py \
    #--testMode 1 --sectionNum 1000 --level 7 --simNum 50  --numRunSample 100000 --makeOneFileOrNot 1
    
    #python3 mainSimpleStep2_sumoSimByFranceParallSim1.py \
    #--testMode 1 --sectionNum 10 --level 7 --simNum 5  --numRunSample 1000 --makeOneFileOrNot 1#测试用
    parser = argparse.ArgumentParser(description="step4D")
    parser.add_argument('-tm','--testMode', default=1, type=int,\
                        help='python多线程测试模式？1为主程序，2为测试多线程,0为不允许多线程模拟')
    
    parser.add_argument('-sn','--sectionNum', default=3000, type=int,\
                        help='#CPU并行运算的要处理的step1例子的分段数')
    
    parser.add_argument('-ll','--level', default=7, type=int,\
                        help='读取step1的例子所在的层数。现在只支持2和7，也就是输出9label和4label')
    
    parser.add_argument('-mk','--makeOneFileOrNot', default=1, type=int,\
                        help='合成一个文件用于后面分析？')
    
    
    parser.add_argument('-su','--simNum', default=15, type=int,\
                        help='每个例子要仿真的次数')
    parser.add_argument('-nrs','--numRunSample', default=18000, type=int,\
                        help='CPU并行运算的要处理的step1例子的数目')
    
    ##################
    '''将输出重定向到文件'''

    fs1 = open('step2_printlog.txt', 'w+')
    sys.stdout = fs1  # 将输出重定向到文件
    
    os.system('rm -rf ./tmpData/*.csv')
    ##################
    
    
    args = parser.parse_args()
    testMode = args.testMode
    sectionNum = args.sectionNum #CPU并行运算的要处理的4B例子的分段数
    level = args.level#读取step4B的例子所在的层数
    makeOneFileOrNot = args.makeOneFileOrNot
    
    simNum = args.simNum #每个例子要仿真的次数
    numRunSample = args.numRunSample #CPU并行运算的要处理的step2例子的数目，数目少可以做测试用
    
     
    ##################
    #print("test1,用于测试测一个样本并进行分析")
    #test1()
    ##################
    #print("test3,运行模拟主程序，用于获得SUMO数据")
    #test3()
    
    
    
    
    ##########################################################
    #主多线程程序
    
    '''print("step2:test4ParallSim1,运行多进程模拟主程序，用于获得SUMO数据,注意SUMO模拟参数现在比较集中")'''
    
    timeST = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if testMode == 1: 
        
        sectionProcess = dict()
        for i in np.arange(sectionNum):
            sectionValue = [i/sectionNum,(i+1)/sectionNum]
            sectionName = "section%d#%3f_%3f#" %(i,sectionValue[0],sectionValue[1])
            
            argsTmp = (sectionValue, sectionName,level, simNum,numRunSample)
            sectionProcess[i] = Process(target=test4ParallSim1, args=argsTmp)
        for i in np.arange(sectionNum):
            sectionProcess[i].start()
            
        for i in np.arange(sectionNum):
            sectionProcess[i].join()
       
    timeED = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

    print("start:%s" %timeST)
    print("end:%s" %timeED)
    
    #############################################################
    #测试多线程
    if testMode == 2:
        p1 = Process(target=test4ParallSim1, args=([0.000,0.01], 'sectionOnly1', ))
        p1.start()
        p1.join()
    
    ############################################################
    if makeOneFileOrNot == 1:#将平行数据转为数据整合一个文件
        
        sectionProcess = dict()
        dfSumoData = pd.DataFrame()
        dfParamsVeh = pd.DataFrame()
        
        for i in np.arange(sectionNum):
            sectionValue = [i/sectionNum,(i+1)/sectionNum]
            sectionName = "section%d#%3f_%3f#" %(i,sectionValue[0],sectionValue[1])
            
            fname ="./tmpData/"+sectionName+"sumoSimData.csv"
            dfSumoDataTmp = pd.read_csv(fname, sep=',')
            dfSumoData = pd.concat([dfSumoData,dfSumoDataTmp])
            
            fname ="./tmpData/"+sectionName+"paramsVehAll.csv"
            dfParamsVehTmp = pd.read_csv(fname, sep=',')
            dfParamsVeh = pd.concat([dfParamsVeh,dfParamsVehTmp])
        
        print(dfSumoData.info())
        print(dfParamsVeh.info())
        
        strTmp = "./data/step2_sumoSimDataLevel%d_simNum%d.csv" %(level,simNum)
        dfSumoData.to_csv(strTmp,index= False)
        
        strTmp = "./data/step2_paramsVehAllLevel%d_simNum%d.csv" %(level,simNum)
        dfParamsVeh.to_csv(strTmp,index= False)
        
if __name__ == "__main__":
    
    main()
    


