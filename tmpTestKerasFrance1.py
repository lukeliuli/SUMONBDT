import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import copy
from sklearn.model_selection import train_test_split




##################################################################
##################################################################


def getKerasModeFloors2(x,enc,saveName):
    model_name = saveName 
    model = keras.models.load_model(model_name)
    yP5= model.predict([x], batch_size=2560)
    nSamples = yP5.shape[0]
     ###需要将预测出的值，转换01整数,并转为数字式
    for i in range(yP5.shape[0]):
        tmp = yP5[i]
        index=  np.argmax(tmp)
        yP5[i] = [0,0,0,0,0]
        yP5[i,index]=1
   

    ###  
    yP5= enc.inverse_transform(yP5)
    yP5= yP5.reshape(-1,nSamples)[0]
    
    yP4 = np.zeros((yP5.shape[0],1))
    yP3 = np.zeros((yP5.shape[0],1))
    yP2 = np.zeros((yP5.shape[0],1))

    for i in range(yP5.shape[0]):
        if(yP5[i]== 2) or (yP5[i]== 1):
             yP4[i] = 21
        else:
             yP4[i] = yP5[i]
                
        if(yP5[i]== 2) or (yP5[i]== 1) or (yP5[i]== 0):
             yP3[i] = 210
        else:
             yP3[i] = yP5[i]
                
        if(yP5[i]== 2) or (yP5[i]== 1) or (yP5[i]== 0) or (yP5[i]== 3):
             yP2[i] = 3210
        else:
             yP2[i] = yP5[i]
    
    return model,yP5,yP4,yP3,yP2

#分层决策树
def dtFitAndSave(x,y,saveName):
    dt = tree.DecisionTreeClassifier(max_depth=7,min_samples_leaf=100)
    dt = dt.fit(x, y)
    tree.plot_tree(dt)
    data=tree.export_graphviz(dt, out_file=None,class_names=None,filled=True) 
    graph = graphviz.Source(data)
    graph.render(saveName)
    
    yPredict = dt.predict(x)
    tmp1 = classification_report(y,yPredict)
    print("纯决策树的识别\n",tmp1)
    mat1num = confusion_matrix(y,yPredict)
    mat2acc = confusion_matrix(y,yPredict,normalize='pred')
    print(mat1num)
    print(np.around(mat2acc , decimals=3))
    #text_representation = tree.export_text(dt)
    #print(text_representation)
    #yPredict = dt.predict_proba(x)
    #index = np.where((yPredict[:,1]<0.98)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    #index = np.where((yPredict[:,1]<0.90)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    #index = np.where((yPredict[:,1]<0.80)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    #index = np.where((yPredict[:,1]<0.70)&(yPredict[:,1]>0.5))
    #print(index[0].shape,index)
    return dt

##########################################################################
##########################################################################
def getDTSamplesInfo(x,dt):
    yPredict = dt.predict_proba(x)
    #print("\n\n getDTSamplesInfo yPredict",yPredict)
    d_path = dt.decision_path(x).todense()
    #print("\n\n d_path",d_path)
    #print("impurity",dt.tree_.impurity)
    #print("feature",dt.tree_.feature)
    #print("threshold",dt.tree_.threshold)
    
    #左节点编号  :  clf.tree_.children_left
    #右节点编号  :  clf.tree_.children_right
    #分割的变量  :  clf.tree_.feature
    #分割的阈值  :  clf.tree_.threshold
    #不纯度(gini) :  clf.tree_.impurity
    #样本个数      :  clf.tree_.n_node_samples
    #样本分布      :  clf.tree_.value
    #https://blog.csdn.net/ywj_1991/article/details/122985778
    #https://www.javaroad.cn/questions/54003
    
    h,w = d_path.shape
    gini =np.zeros((h,1))
    
    
    
    for i in range(h):
       path = d_path[i]
       v,ind = np.where(path>0)
       xtmp = x[i]
       #print("path",path,ind,np.array(ind)[-1])
    
       #print("\n index",index)
       #print("impurity",dt.tree_.impurity[ind])
       #print("feature",dt.tree_.feature[ind])
       #print("threshold",dt.tree_.threshold[ind])
       #print("x[index]",xtmp[ind])
       
      
       #print("the leaf node:",np.array(ind)[-1],"the simplest rule is")
       #for jj in ind:
       #    if dt.tree_.feature[jj] == -2:
       #         print("label,proba is",yPredict[i,0],yPredict[i,1])
       #         break
                
       #    if xtmp[jj]<=dt.tree_.threshold[jj]:
       #       print(" x[%d]<=%.3f" %(dt.tree_.feature[jj],dt.tree_.threshold[jj]))
       #    else:
       #       print(" x[%d]>%.3f" %(dt.tree_.feature[jj],dt.tree_.threshold[jj]))
                    
       finalPos = np.array(ind)[-1]
       gini[i] = dt.tree_.impurity[finalPos]
       
       #print("d_path",i,path,dt.tree_.impurity[finalPos])
       #print(dt.tree_.feature[finalPos])
       #print(dt.tree_.threshold[finalPos])
       #print(dt.tree_.n_node_samples[finalPos])

    
       return gini,yPredict


##########################################################################
from tqdm import tqdm
def computeAndCompareHybridMode(x,y,dt,kerasPLabel,floorLabel):
    nSamples,feturesNume  = x.shape
    yHyLabel  = np.zeros((nSamples,1))
    giniFloor,yPredictProFloor = getDTSamplesInfo(x,dt)
    prdictMax = np.max(yPredictProFloor,axis=1)
    
    
    index1 = np.argmax(yPredictProFloor, axis = 1)
    index1 = index1.astype('int64')
    hyCounter = nSamples
    for i in tqdm(range(nSamples)):
        yHyLabel[i] = floorLabel[index1[i]]
        giniTmp = giniFloor[i]
        probaTmp = prdictMax[i]
        if giniTmp>0.1 or probaTmp<0.95:
            yHyLabel[i] = kerasPLabel[i]
            hyCounter = hyCounter-1
        

    print("混合识别的结果\n")
    print('floorLabel\n',floorLabel) 
    print('hyCounter\n',hyCounter)    


    tmp1 = classification_report(y,yHyLabel)
    print('hybrid\n',tmp1)
    tmp1 = classification_report(y,kerasPLabel)
    print('keras\n',tmp1)
    mat1num = confusion_matrix(y,yHyLabel)
    mat2acc = confusion_matrix(y,yHyLabel,normalize='pred')
    print('mat1num\n',mat1num)
    print('mat2acc\n',np.around(mat2acc , decimals=3))
    return
'''
from tqdm import tqdm

def computeAndCompareHybridMode(x,y,dt,kerasPLabel,floorLabel):
    nSamples,feturesNume  = x.shape
    yHyLabel  = np.zeros((nSamples,1))
    hyCounter = 0
    for i in tqdm(range(nSamples)):
        #print(i)
        xtmp = x[i]
        giniFloor,yPredictProFloor = getDTSamplesInfo([xtmp],dt)
        giniTmp = giniFloor[0]
        yPredictProFloorTmp = yPredictProFloor[0]
        #print('gini',giniTmp )
        #print('probPredict',yPredictProFloor0Tmp  )
        if giniTmp >0.05 or max(yPredictProFloorTmp)<0.98:
            yHyLabel[i] = kerasPLabel[i]
        else:
            #floorLabel= [3,4,210]
            index = np.argmax(yPredictProFloorTmp)
            yHyLabel[i] = floorLabel[index]
            hyCounter = hyCounter+1
    print('floorLabel\n',floorLabel) 
    print('hyCounter\n',hyCounter)    


    tmp1 = classification_report(y,yHyLabel)
    print('hybrid\n',tmp1)
    tmp1 = classification_report(y,kerasPLabel)
    print('keras\n',tmp1)
    mat1num = confusion_matrix(y,yHyLabel)
    mat2acc = confusion_matrix(y,yHyLabel,normalize='pred')
    print('mat1num\n',mat1num)
    print('mat2acc\n',np.around(mat2acc , decimals=3))
    return
'''
##########################################################################
###简单模型2，有隐藏层
def kerasFitAndSaveSimple2(x,yOneHot,num_labels,filename):
    nSamples,features_size = x.shape
    relu_size = 512
    dropout_rate = 0.05
    build_model = tf.keras.Sequential()
    build_model.add(layers.Dense(relu_size, activation='relu',name="layer1",input_shape=(features_size,)))
    build_model.add(layers.Dropout(dropout_rate,name="Dropout1-2"))
    build_model.add(layers.Dense(relu_size/2, activation='relu',name="layer2"))
    build_model.add(layers.Dropout(dropout_rate,name="Dropout2-3"))
    build_model.add(layers.Dense(num_labels, activation='sigmoid',name="layer3"))
    #model = tf.keras.Model(inputs=[features], outputs=[build_model])
    #enc = OneHotEncoder()
    #enc.fit(y)  
    #yOnehot=enc.transform(y).toarray()
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    
    build_model = keras.models.load_model(filename)
    
    build_model.fit([x],[yOneHot],epochs=30000, batch_size=10000*1)
    #build_model.fit(x,yOneHot,epochs=1000, batch_size=80000*1)
    #build_model.save("kerasSimple2.h5")
    build_model.save(filename)
    plot_model(build_model, to_file='KerasSimple2_HiddenLayer.png', show_shapes=True)
    return build_model


###############################################################
###简单模型3，resnet_like

def global_model(dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    return model

def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid',name="global"))
    return model

def kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName):
    nSamples,features_size = x.shape
    relu_size = 512
    dropout_rate = 0.05
    hierarchy = [1,1,1]
    global_models = []
    label_size = num_labels
    features = layers.Input(shape=(features_size,))
    for i in range(len(hierarchy)):
        if i == 0:
            global_models.append(global_model(dropout_rate, relu_size)(features))
        else:
            global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))

    p_glob = sigmoid_model(label_size)(global_models[-1])
    build_model = tf.keras.Model(inputs=[features], outputs=[p_glob])
    #model = tf.keras.Model(inputs=[features], outputs=[build_model])
    #enc = OneHotEncoder()
    #enc.fit(y)  
    #yOnehot=enc.transform(y).toarray()
    
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    
    build_model = keras.models.load_model(saveName)
    build_model.fit([x],[yOneHot],epochs=10000, batch_size=10000*1)
    #saveName = "KerasSimple3_likeResnet.h5"
    build_model.save(saveName)
    #plot_model(build_model, to_file='KerasSimple3_likeResnet.png', show_shapes=True)
    return build_model

def getKerasResnetRVL(x,enc,saveName):
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
####################################################################################################
####用法国数据进行验证
print("用法国数据进行验证,训练次数为30000，大概18小时")
file1 = "./trainData/france_0_allSamples.csv"
print("reading data france")
xyDataTmp = pd.read_csv(file1)
#print(xyDataTmp.info())
xyData = np.array(xyDataTmp)
h,w = xyData.shape
x = xyData[:,1:23]#简单处理与SUMO数据库一致
x = xyData[:,1:w-1]#用所有的数据
y = xyData[:,w-1]
y= y.astype('int64')
yl5 = y
print("x.shape:",x.shape,"y.shape:",y.shape,"y.type:", type(y) )
del xyDataTmp #节省内存
del xyData #节省内存


####
'''
file1 = "./trainData/dataAllSim1000.csv"
print("reading data")
xyDataTmp = pd.read_csv(file1)
#print(xyDataTmp.info())
xyData = np.array(xyDataTmp)

xSumo = xyData[:,0:22]
ySumo = xyData[:,22:26]
ySumo= ySumo[:,2]#01234
ySumo= ySumo.astype('int64')

print("x.shape:",x.shape,"yl5.shape:",yl5.shape)
del xyDataTmp #节省内存
del xyData #节省内存


x = np.concatenate((xSumo,x))
y = np.concatenate((ySumo,y))
yl5 = y
print("x.shape:",x.shape,"y.shape:",yl5.shape,"y.type:", type(yl5) )
'''

##########################################################################
###keras拟合,oneHot
nSamples,nFeatures =  x.shape
enc = OneHotEncoder()
yl5= yl5.reshape(nSamples,-1)
enc.fit(yl5)  


##keraskeras拟合
filename = "kerasSimple2FranceDataSetAll1.h5"
if 0:
    x_train, x_test, y_train, y_test = train_test_split(x, yl5, test_size = 0.5)
    yOneHot=enc.transform(y_train).toarray()
    num_labels = 5 
    simpleMode2 = kerasFitAndSaveSimple2(x_train,yOneHot,num_labels,filename)

filename = "KerasSimple3_likeResnet_floor4_5label.h5"
if 1:
    x_train, x_test, y_train, y_test = train_test_split(x, yl5, test_size = 0.5)
    
    yOneHot=enc.transform(yl5).toarray()
    num_labels = 5 
    kerasModel3_floor4_5label = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,filename) 
##########################################################################



index = np.where((yl5 == 2) | (yl5 == 1))
yl4 = yl5.copy()
yl4[index]=21
#print(yl4)


index = np.where((yl5 == 2) | (yl5 == 1) | (yl5 == 0))
yl3 = yl5.copy()
yl3[index]=210
#print(yl3)



index = np.where( (yl5 == 3)|(yl5 == 2) | (yl5 == 1) | (yl5 == 0))
yl2 = yl5.copy()
yl2[index]=3210
#print(yl2)

##########################################################################

#hierachFloor['floor3']["dt"] = dtFitAndSave(x,yl2,"Floo3_2")
#hierachFloor['floor2']["dt"] = dtFitAndSave(x,yl3,"Floo2_3")
#hierachFloor['floor1']["dt"] = dtFitAndSave(x,yl4,"Floor1_4")

#dt_floor1_2label = dtFitAndSave(x,yl2,"Floor1_2label")
#dt_floor2_3label = dtFitAndSave(x,yl3,"Floor2_3label")
#dt_floor3_4label = dtFitAndSave(x,yl4,"Floor3_4label")
dt_floor4_5label = dtFitAndSave(x,yl5,"Floor4_5label")
kerasFloors,yKerasP5,yKerasP4,yKerasP3,yKerasP2=getKerasModeFloors2(x,enc,filename)


#floor=3 ,label=2
#dt = hierachFloor['floor3']["dt"]

#floorLabel= [4,3210] 
#computeAndCompareHybridMode(x,yl2,dt,yKerasP2,floorLabel)



dt =dt_floor4_5label
floorLabel= [0,1,2,3,4] 
computeAndCompareHybridMode(x,yl5,dt,yKerasP5,floorLabel)