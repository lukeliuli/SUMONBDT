#用于简单过路灯模拟
#1.单直车道. 2.无车道转换。3.所有车假设为同一类车，也就是汽车动力学和汽车运动学一样
#作者lukeliuli@163.com
# -*- coding: utf-8 -*-
#https://sumo.dlr.de/pydoc/traci.html
"""
@author: lukeliuli@163.com
"""


import numpy as np
import os
import sys
import random

#import traci as libsumo
import libsumo
def simSumoCmdGreenTls(params):
    #print(params)
    simNum =  params["simNum"] #= 150
    greenTime =  params["greenTime"]# = 15
    otherVehs =   params["otherVehs"] #= [[1, 0],[7,0],[14,0]]  # [距离交通灯的距离1，行驶速度1,距离交通灯的距离2，行驶速度2]
    otherVehsParams =   params["otherVehsParams"] #= [5.5, 2, -9, 60 /3.6, 0.2]  # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距]
    objectVeh =  params["objectVeh"] #= [50, 0] 距离交通灯的距离,当前速度
    objectVehParams =  params["objectVehParams"] #= [5.5, 2, -9, 60 /3.6, 0.2]  # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距]
    
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
    #path2 = path+"\sumoCfgs\my1Lane1Tls-server.sumocfg"#window
    path2 = path+"/sumoCfgs1/my1Lane1Tls-server.sumocfg"#linux
    sumoBinary = "sumo"
    
    
    #libsumo.start(["sumo-gui", "-c", path2])
    libsumo.start([sumoBinary, "-c", path2,'--ignore-route-errors'])
    
    #####基本车辆参数设定
    
    
    
    
    
      
                            
                          
     # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                               
    length,maxAcc,maxDacc,maxSpeed,tau,minGap,imperfection,speedFactor = objectVehParams
    libsumo.vehicletype.setAccel(typeID1,maxAcc)  
    libsumo.vehicletype.setDecel(typeID1,maxDacc) 
    libsumo.vehicletype.setImperfection(typeID1,imperfection)
    libsumo.vehicletype.setLength(typeID1,length) 
    libsumo.vehicletype.setMaxSpeed(typeID1,maxSpeed)   
    libsumo.vehicletype.setMinGap(typeID1,minGap) 
    libsumo.vehicletype.setSpeedFactor(typeID1,speedFactor)  
    libsumo.vehicletype.setTau(typeID1,tau)  
  
                          
                            
    # [车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,速度噪声]                         
    length,maxAcc,maxDacc,maxSpeed,tau,minGap,imperfection,speedFactor = otherVehsParams
    libsumo.vehicletype.setAccel(typeID2,maxAcc)  
    libsumo.vehicletype.setDecel(typeID2,maxDacc) 
    libsumo.vehicletype.setImperfection(typeID2,imperfection)
    libsumo.vehicletype.setLength(typeID2,length) 
    libsumo.vehicletype.setMaxSpeed(typeID2,maxSpeed)   
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
    
    lgc1.phases[0].state = 'y'
    lgc1.phases[0].duration =yellowDurTime
    lgc1.phases[1].state = 'G'
    lgc1.phases[1].duration =greenTime#修改绿灯时间
    
    lgc1.phases[2].state = 'y'
    lgc1.phases[2].duration =yellowDurTime
    
    lgc1.phases[3].state = 'r'
    lgc1.phases[3].duration =redDurTime
    
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
    while requireStop == 0:
        if (stepNum == 0):
            dist,vel = objectVeh
            libsumo.vehicle.add(objvehID, routeID, typeID=typeID1, depart='now', departLane='first', \
                        departPos=trafficLightPos-dist, departSpeed=str(vel))
            counter = 0
            for v in otherVehs:
                counter = counter+1
                dist,vel = v
                libsumo.vehicle.add(otherVehID+str(counter), routeID, typeID=typeID2, depart='now', departLane='first', \
                                    departPos=str(trafficLightPos-dist), departSpeed=str(vel))
        
        libsumo.simulationStep()
        stepNum += 1

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




        if timeT/1000 > 30 or states == "r" or dist<=0:  # if time is over 30 second stoping the simulation
            requireStop = 1


        #print("#tls state: %s,duration:%.3f,running time(ms):%d, \
        #passTLS:%d,lanePos:%s,dist2TLS:%d" % (states,phaseDur,timeT,passTLS,vehLanePos,dist2TLS))

    libsumo.close()
    return None

############################################################################################
############################################################################################                                   ############################################################################################                

############################################################################################
###测试程序1
#定义模拟过程中动态可调整参数
params =dict()
params["simNum"] = 1
params["greenTime"] = 15
params["otherVehs"] = [[1, 0],[7,0],[14,0]]  # [[距离交通灯的距离1，行驶速度1],[距离交通灯的距离2，行驶速度2]]
                            
#[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                              
params["otherVehsParams"] = [5,2,-2,60/3.6,0.5, 0.3 ,0.1,0.05] 
                            
params["objectVeh"] = [30,0]
#[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]  
params["objectVehParams"] = [5,2,-2,60/3.6,0.5, 0.3 ,0.1,0.05] 
#for i in range(params["simNum"]):
    #data = simSumoCmdGreenTls(params)


############################################################################################
###测试程序2
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
import numpy as np

####获得一个样本进行测试
df1 = pd.read_csv('./trainData/france_0_allSamples.csv')
df1.head(5)
df2 = df1.loc[df1['redLightTime']>20]
df2 = df2.loc[df1['speedFlag'] ==4]
df2 = df2.loc[df1['vehPos_10'] >0]
tmp = df2.iloc[0].values
#print(tmp)
vehID,redLightTime,distToRedLight,speed,laneAvgSpeed,arriveTime1,arriveTime2,\
    vehPos_1,vehSpeed_1,vehPos_2,vehSpeed_2,vehPos_3,vehSpeed_3,vehPos_4,vehSpeed_4,vehPos_5,vehSpeed_5,\
    vehPos_6,vehSpeed_6,vehPos_7,vehSpeed_7,vehPos_8,vehSpeed_8,vehPos_9,vehSpeed_9,vehPos_10,vehSpeed_10,\
    vehPos_11,vehSpeed_11,vehPos_12,vehSpeed_12,vehPos_13,vehSpeed_13,vehPos_14,vehSpeed_14,vehPos_15,vehSpeed_15,\
    vehPos_16,vehSpeed_16,vehPos_17,vehSpeed_17,vehPos_18,vehSpeed_18,vehPos_19,vehSpeed_19,vehPos_20,vehSpeed_20,\
    speedFlag = tmp
vehObj = np.array([distToRedLight,speed])
vehsOthers = tmp[7:-1]
vehsOthers = vehsOthers.reshape(-1,2)
print(vehObj)
print(vehsOthers)
vehsOthers = vehsOthers[np.where(vehsOthers[:,0]>0)]
print(vehsOthers)
greenTime = redLightTime

#####################################################################
params =dict()
params["simNum"] = 100
params["greenTime"] = greenTime
params["otherVehs"] = vehsOthers  # [[距离交通灯的距离1，行驶速度1],[距离交通灯的距离2，行驶速度2]]
                            
#[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]                              
params["otherVehsParams"] = [4.5,2,9,60/3.6,0.5, 0.8 ,0.01,0.05] 
                            
params["objectVeh"] = vehObj
#[车辆长度，最大加速度,最大减加速度，最大速度，反应时间,最小间距,不专心,速度噪声]  
params["objectVehParams"] = [4.5,2,9,60/3.6,0.5, 0.8,0.01,0.05] 

for i in range(params["simNum"]):
    data = simSumoCmdGreenTls(params)

