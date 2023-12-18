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
print("基于SUMO和蒙特卡洛模拟分析生成预测当前样本的最小速度，原始对应程序为mainSimSumoFranceData，需要在docker的原始环境下运行，没有conda")
print("程序为0.1")


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
########################################################################################################################

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
                        departPos=trafficLightPos-dist, departSpeed=str(vel))
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
        meanSpeed = libsumo.lane.getLastStepMeanSpeed(laneID)
        laneMaxSpeed = libsumo.lane.getMaxSpeed(laneID)

        vehMaxSpeed = min(vehMaxSpeed, laneMaxSpeed)

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
###测试程序1，测试直接从france数据库的提取样本
def test1():
    ####获得一个样本进行测试
    df1 = pd.read_csv('./trainData/france_0_allSamples1.csv')
    #df1.head(5)
    df2 = df1.loc[df1['redLightTime']>10]
    df2 = df2.loc[df1['speedFlag'] ==3]
    df2 = df2.loc[df1['vehPos_3'] >0]



    randIndex = random.randint(0,len(df2))
    tmp = df2.iloc[randIndex].values


    vehID,redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2,numStillVeh,ArrivalDivRedTime,\
        vehPos_1,vehSpeed_1,vehPos_2,vehSpeed_2,vehPos_3,vehSpeed_3,vehPos_4,vehSpeed_4,vehPos_5,vehSpeed_5,\
        vehPos_6,vehSpeed_6,vehPos_7,vehSpeed_7,vehPos_8,vehSpeed_8,vehPos_9,vehSpeed_9,vehPos_10,vehSpeed_10,\
        vehPos_11,vehSpeed_11,vehPos_12,vehSpeed_12,vehPos_13,vehSpeed_13,vehPos_14,vehSpeed_14,vehPos_15,vehSpeed_15,\
        vehPos_16,vehSpeed_16,vehPos_17,vehSpeed_17,vehPos_18,vehSpeed_18,vehPos_19,vehSpeed_19,vehPos_20,vehSpeed_20,\
        speedFlag = tmp
    vehObj = np.array([distToRedLight,speed])
    vehsOthers = tmp[9:-1]
    vehsOthers = vehsOthers.reshape(-1,2)
    vehsOthers_all = vehsOthers[np.where(vehsOthers[:,0]>0)]
    vehsOthers1 = vehsOthers_all[0:-1]
    print("vehObj:",vehObj)
    print("vehsOthers1",vehsOthers1)
    print("vehs_all",vehsOthers_all)#数据命名错误，vehsOthers等于车道上所有车
    
    #print(vehsOthers)
    #redLightTime = redLightTime



    time.sleep(5);
    #####################################################################
    params =dict()
    params["simNum"] = 50
    params["redLightTime"] = redLightTime
    params["otherVehs"] = vehsOthers1  # [[距离交通灯的距离1，行驶速度1],[距离交通灯的距离2，行驶速度2]]

    #[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                              
    params["otherVehsParams"] = [5,2,9,60/3.6,     0.5, 0.5 ,0.01,0.05] 

    params["objectVeh"] = vehObj
    #[车辆长度，最大加速度,最大减加速度，最大速度，反应时间(0.01到0.1的传输延迟，0.2到0.5的执行延迟),最小间距,不专心,速度噪声]  
    params["objectVehParams"] = [5,2,9,60/3.6,                       0.5,                           0.5,      0.01,  2] 
    params["log"] = False
    minSpeedList = []
    
    for i in range(params["simNum"]):

         #加入噪声
         params["otherVehsParams"] = [5,2+random.uniform(0,1),9,60/3.6,0.3+random.uniform(0,0.1), 0.5 ,0.01,2] 
         params["objectVehParams"] = [5,2+random.uniform(0,1),9,60/3.6,0.3+random.uniform(0,0.1), 0.5, 0.01,2] 
         statRec1,strRec1,minSpeed,leaderInfo = simSumoCmd(params)


         print("\nsimNum:%d" %i)
         print(strRec1)
         print("leaderInfo",leaderInfo )
         print('minSpeed',minSpeed)
         minSpeedList.append(minSpeed)

    minSpeedList1 = np.array(minSpeedList)
    print("minSpeedList ,min:%.2f,max:%.2f,mean:%.2f" %(np.min(minSpeedList1),np.max(minSpeedList1),np.mean(minSpeedList1)))
    speedSlot = ["[0,5/3.6]","[5/3.6,15/3.6]","[15/3.6,25/3.6]","[25/3.6,35/3.6]","[35/3.6,80/3.6]"]
    print("origin speedFlag %d,speedSlot %s\n\n" %(speedFlag,speedSlot[speedFlag]))
    plt.hist(minSpeedList)
 
########################################################################################################################
###测试程序2,从pickle中读取样本,注意与france csv不一样，输入数据只有48，把vehID,speedFlag 已经去掉了
import pickle



def configAndRun(tmp,index):
    
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
    print("而模拟中车辆的启动时间很快，能1秒内加速到2m/s，所以模拟中实际红灯时间加1.5秒")
    if vehsOthers_all[0][1] <1/3.6 and vehsOthers_all[0][0] >0:#头车速度低于1km/h
        redLightTime= redLightTime+1.5
    if vehsOthers_all.shape[0]>1  and vehsOthers_all[1][1] <1/3.6 and vehsOthers_all[1][0] >0:#存在第二步车，而且速度低于1km/h
        redLightTime= redLightTime
    print("redLightTime",redLightTime)
    
    #time.sleep(5);
    #####################################################################
    
    params =dict()
    params["simNum"] = 100
    
    params["otherVehs"] = vehsOthers1  # [[距离交通灯的距离1，行驶速度1],[距离交通灯的距离2，行驶速度2]]

    #[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                              
    params["otherVehsParams"] = [5,2,9,60/3.6,     0.5, 0.5 ,0.01,0.05] 

    params["objectVeh"] = vehObj
    #[车辆长度，最大加速度,最大减加速度，最大速度，反应时间(0.01到0.1的传输延迟，0.2到0.5的执行延迟),最小间距,不专心,速度噪声]  
    params["objectVehParams"] = [3,2,9,60/3.6,                       0.5,                           0.5,      0.01,  0.05] 
    params["log"] = False
    minSpeedList = []
    
    paramsVehList = []
    for i in range(params["simNum"]):
         #print("\nsimNum:%d Start" %i)
         #加入噪声
         params["otherVehsParams"] = [3,1+random.uniform(0,1),9,15/3.6+random.uniform(0,45/3.6), \
                                      0.1+random.uniform(0,0.4), 0.1+random.uniform(0,0.3) ,0.00,0.00] 
         params["objectVehParams"] = [3,1+random.uniform(0,1),9,15/3.6+random.uniform(0,45/3.6), \
                                      0.1+random.uniform(0,0.4), 0.1+random.uniform(0,0.3) ,0.00,0.00] 
      
        
        
         #随机0.5秒为驾驶员的反应时间和车辆启动时间的附加随机值
         params["redLightTime"] = float(redLightTime+random.uniform(0.0,0.5))
            
         statRec1,strRec1,minSpeed,leaderInfo,requireStop  = simSumoCmd(params)
     

        #print("simNum:%d End" %i)
        #print(strRec1)
        #print("leaderInfo",leaderInfo )
        #print('minSpeed',minSpeed)
        #print('requireStop',requireStop)

         if requireStop<0:
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
        print("MonteCarloSimulation minSpeedList ,min:%.2f,max:%.2f,mean:%.2f" %(np.min(minSpeedList1),np.max(minSpeedList1),np.mean(minSpeedList1)))
        
   
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


#test1()#测试程序1，测试直接从france数据库的提取样本

########################################################################################################################
#test2 #测试程序2,从pickle中读取样本，样本已经进行[x,y]分割,样本为原始的1%"
'''
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import math
def test2():
    print("运行test2,测试程序2,从pickle中读取样本，样本已经进行[x,y]分割.样本为2slot,5hier,9label\n\n")
    fpk=open('samples2.pkf','rb')   
    [xFloors,yFloors,modSaveNameFloors,encLevels,yKerasFloors]=pickle.load(fpk)  
    fpk.close()   
    
    
    
    dataFile=open('sumoSimData15000.csv','w+',buffering=1)
    strData = "floor,numSamples,index,outputAvgSpeed,outputY,sumoOutput,yKerasOutput"
    print(strData,file=dataFile)
    
    logFile=open('log.txt','w+',buffering=1)
    strLog = "start time %s" %datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(strLog,file=logFile)
    
  
    
    #for i in [len(xFloors)-1]:
    indexSample = 0
    for i in range(len(xFloors)):
        levelIndex = i
        x = xFloors[str(i)]
        yCurLayer1 =  yFloors[str(i)]
        yKeras  = yKerasFloors[str(i)]
        
        
        #欠采样
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        print(sorted(Counter(y_resampled).items()))
    
        sampleNum =  min(15000,math.floor(yCurLayer1.shape[0]/10))#样本为原始的1%,100样本为3分钟
        
        timeNow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        strLog = "\n floor:%d,sampleNum:%d,start time %s" %(i,sampleNum,timeNow)
        print(strLog,file=logFile)
    
        for j in range(sampleNum):
            #sample_name = 1(ID)+8(keyFeature)+40(otherVehcle)+6(keyFeatures)+40(otherVehs)+1(flag)= 96
            #x-name = 8(keyFeature)+40(otherVehcle)+6(keyFeatures)+40(otherVehs)= 94
            tmp = x[j][0:48]
           
            minSpeedList1 = configAndRun(tmp)
            
            if len(minSpeedList1) == 0:
                continue
            
        
            outputAvgSpeed = np.round(np.mean(minSpeedList1),2)
            outputSpeedTag = minSpeed2Tag(outputAvgSpeed)
            outputY = yCurLayer1[j]
            #print(outputY,type(outputY))
            outputY = outputY[0].tolist()
          
            strData = "%d,%d,%d,%.2f,%s,%s,%s" %(i, \
                                                 sampleNum,\
                                                 j,\
                                                 outputAvgSpeed, \
                                                 outputY, \
                                                 str(outputSpeedTag), \
                                                 yKeras[j])
            print(strData,file=dataFile)
            
            
            
    timeNow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strLog = "\n end time %s" %timeNow
    print(strLog,file=logFile)
    logFile.close() 
    dataFile.close() 
       
'''    
    
########################################################################################################################
#test3
from datetime import datetime

import math
def test3():
    print("运行test3,测试程序2,从lowprobSamples.pkf，样本已经进行[x,y]分割.样本为2slot,5hier,9label\n\n")
    fpk=open('lowprobSamples.pkf','rb')   
    [xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN]=pickle.load(fpk)  
    fpk.close()   
    
    
    logFile=open('log.txt','w+',buffering=1)
    strLog = "start time %s" %datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(strLog,file=logFile)
    
    numSamples,numFeatures = xlowpra.shape


    rvl = []
    #numSamples = 2000
    paramsVehAll = []
    for j in range(0,numSamples):
       
        #xlowpra:x-name = 8(keyFeature)+40(otherVehcle)+7(keyFeatures)+39(otherVehs)= 94
        print("#################################sampleNum:",j)
        tmp = xlowpra[j][0:48]

        minSpeedList1,paramsVehList = configAndRun(tmp,j)

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
       
        if j%100 ==5:
            df = pd.DataFrame(rvl)
            fs = "sumoSimDataTmp.csv"
            #[5+9+2+8]
            #车辆长度，最大加速度,最大减加速度，最大速度，反应时间(0.01到0.1的传输延迟，0.2到0.5的执行延迟),最小间距,不专心,速度噪声
            #'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor'
            df.to_csv(fs,index= False, header=['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                               'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                               'smv1','smv2'])
            df = pd.DataFrame(paramsVehAll)
            fs = "paramsVehAllTmp.csv"
            df.to_csv(fs,index= False, header=['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                       'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                               'minSpeed0'])
            
            
        timeNow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        strLog = "\nsampleIndex:%d, time %s" %(j,timeNow)
        print(strLog,file=logFile)
     
    df = pd.DataFrame(rvl)
    fs = "sumoSimData%d.csv" %j
    #[5+2+9]
    df.to_csv(fs,index= False, header=['sampleIndex','outputAvgSpeed','originOutput','sumoOutputSpeedTag','kerasPredictLabel',\
                                               'NN0','NN1','NN2','NN3','NN4','NN5','NN6','NN7','NN8',\
                                               'smv1','smv2'])
    
    df = pd.DataFrame(paramsVehAll)
    fs = "paramsVehAll%d.csv" %j
    df.to_csv(fs,index= False, header=['sampleIndex','vehLen0','maxAcc0','maxDAcc0','maxSpeed0','reacTime0','minGap0','Impat0','speedFactor0',\
                                       'vehLen','maxAcc','maxDAcc','maxSpeed','reacTime','minGap','Impat','speedFactor',\
                                               'minSpeed0'])
                                  
    logFile.close() 
 
######################################################################################################################## 
########################################################################################################################   
########################################################################################################################


def main():
    test3()
if __name__ == "__main__":
    main()
    


