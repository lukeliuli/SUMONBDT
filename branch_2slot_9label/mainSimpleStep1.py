
########################################################################################################################
print("1.接编号为0的主程序,先找出低概率样本，")
print("2.对较低概率的样本进行蒙特卡洛模拟分析，原始对应程序为mainSimSumoFranceDatra")
print("3.最终进行分析，程序编号为1")
print("3.最终进行分析，有简化程序mainSimpleStep1.py")
########################################################################################################################
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import tree
#import graphviz 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import copy
from imblearn.over_sampling import RandomOverSampler
import pickle

import warnings
warnings.filterwarnings("ignore")
import argparse      

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
'''print("\n2slot的数据列表为：headName2SlotXY96\n")'''


def main():
    
    

    ########################################################################################################################
    print("1.1 主程序开始")
    ########################################################################################################################
    #python3 mainSimpleStep1.py --level 0 1 2 3 4 5 6 7 --threshold 0.7
    parser = argparse.ArgumentParser(description="step1")
    parser.add_argument('-ll','--level', default=[0,1,2,3,4,5,6,7], nargs='+',type=int,help='step0 读取训练好的模型第几layers?，就获取所有hierarchy=[0,1,2,3,4,5,6,7]')
    parser.add_argument('-th','--threshold', default=0.7, type=float,help='取测试样本中可能性最大值小于0.7的例子')
  
    args = parser.parse_args()
    level = args.level
    threshold= args.threshold #CPU并行运算的要处理的4B例子的分段数
    
    
    ########################################################################################################################
    #####用现有训练模型进行预测
    fpk=open('step0_sepTrainedSamplesAll_8level.pkf','rb') #step0_sepTrainedsSamplesAll_8level.pkf
    [xFloors,yFloors,modSaveNameFloors,encLevels,xTestFloors, yTestFloors]=pickle.load(fpk)  
    fpk.close()   

    fpk=open('step0_sepTestedSamplesAll_8level.pkf','rb') 
    [xFloors,yFloors,modSaveNameFloors,encLevels,yKerasFloors,xTestFloors, yTestFloors]=pickle.load(fpk)  
    fpk.close()  

    
    
    for i in level:#i=2 ,对应输出4个类别
    
            levelIndex = i 
            x = xTestFloors[str(i)]
            yCurLayer1 =  yTestFloors[str(i)]
            #yP = yKerasFloors[str(i)]
            numLayers = 4
            modeSaveName = "../trainedModes/modelSep-9level%d-%dlayer-2slots-gpu1.h5" %(i,numLayers)
            model = keras.models.load_model(modeSaveName)
            yPredictOut= model.predict([x], batch_size=2560)#预测并将onehot转为label
            yPredictOut = np.around(yPredictOut , decimals=3)
            print('yPredictOut.shape:',yPredictOut.shape)
            ymax1=np.max(yPredictOut,axis=1)#将onehot转为label.提取最大概率
            ymax2=np.argmax(yPredictOut,axis=1)#将onehot转为label

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            index = np.where(ymax1<threshold)[0]#         取最大值小于0.7的例子                                                                                                                                                                        

            ylowpraPredictNN=yPredictOut[index]#对较低概率的样本
            xlowpra=x[index]
            ylowpraLabel = yCurLayer1[index]
            ylowPredictLabel = ymax2[index].reshape(-1,1)


            print("xlowpra.shape",xlowpra.shape)
            strTmp = 'step1_lowprobSamplesLevel%d.pkf' %i
            fpk=open(strTmp,'wb+') 
            pickle.dump([xlowpra,ylowpraLabel,ylowPredictLabel,ylowpraPredictNN],fpk)  
            fpk.close() 
        
   
        

       

      
if __name__=="__main__":
    main()

