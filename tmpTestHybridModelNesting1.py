#############################################################
print("建立多层嵌套决策树模型")
#############################################################
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


###############################################################

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
    
    #build_model = keras.models.load_model(saveName)
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
##########################################################################
def hybridTest(x,y,dt,floorLabel,kerasLabel):
    nSamples,feturesNume  = x.shape
    yHyLabel  = np.zeros((nSamples,1))#混合模型预测标签
    dtLabel = np.zeros((nSamples,1))#决策树模型预测标签
    gini,yPredictProb= getDTSamplesInfo(x,dt)
    prdictMax = np.max(yPredictProb,axis=1)
    

    
    index1 = np.argmax(yPredictProb, axis = 1)
    index1 = index1.astype('int64')
    hyCounter = nSamples
    for i in tqdm(range(nSamples)):
        yHyLabel[i] = floorLabel[index1[i]]
        dtLabel[i] = floorLabel[index1[i]]
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
    tmp1 = classification_report(y,kerasLabel)
    print('keras\n',tmp1)
    mat1num = confusion_matrix(y,yHyLabel)
    mat2acc = confusion_matrix(y,yHyLabel,normalize='pred')
    print('mat1num\n',mat1num)
    print('mat2acc\n',np.around(mat2acc , decimals=3))
    return



##############################################################################################################################
#############################################################
print("0.主程序开始，建立多层嵌套决策树模型")
#############################################################
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
xOrigin =x
print("x.shape:",x.shape,"y.shape:",y.shape,"y.type:", type(y) )
del xyDataTmp #节省内存
del xyData #节省内存



##################################
print("原始样本分为3210和4两类")
index0 = np.where( (yl5 == 3)|(yl5 == 2) | (yl5 == 1) | (yl5 == 0))
index1 = np.where( (yl5 ==4))
x3210_4 = xOrigin.copy()
y3210_4 = yl5.copy()
y3210_4[index0] = 3210



##################################
print("3210样本分为210和3两类")
index0 = np.where( (yl5 == 2) | (yl5 == 1) | (yl5 == 0))
index1 = np.where( (yl5 ==3))
xTmp = xOrigin[index0]
yTmp = yl5[index0]
yTmp[:] = 210



x = np.concatenate((xTmp,xOrigin[index1]),axis=0)
y = np.concatenate((yTmp,yl5[index1]),axis=0)
x210_3 = x
y210_3 = y
print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
print(y210_3)

##################################
print("210样本分为10和2两类")
index0 = np.where((yl5 == 1) | (yl5 == 0))
index1 = np.where( (yl5 ==2))
xTmp = xOrigin[index0]
yTmp = yl5[index0]
yTmp[:] = 10

x = np.concatenate((xTmp,xOrigin[index1]),axis=0)
y = np.concatenate((yTmp,yl5[index1]),axis=0)
x10_2 = x
y10_2 = y
print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
print(y10_2)


##################################
print("10样本分为0和1两类")
index0 = np.where( (yl5 == 0))
index1 = np.where( (yl5 ==1))
xTmp = xOrigin[index0]
yTmp = yl5[index0]

x = np.concatenate((xTmp,xOrigin[index1]),axis=0)
y = np.concatenate((yTmp,yl5[index1]),axis=0)

x0_1 = x
y0_1 = y
print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
print(y0_1)

if 1:
    x=xOrigin
    y=yl5
    print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
    
    num_labels = 5 
    nSamples,nFeatures =  x.shape
    enc = OneHotEncoder()
    y= y.reshape(nSamples,-1)
    
    print("y.shape:",y .shape,"y.type:", type(y) )
    enc.fit(y)
    yOneHot=enc.transform(y).toarray()
    saveName = "hybrid2_KerasSimple3_likeResnet_5label.h5"
    kerasModel3_5label = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
    yKeras_5label=getKerasResnetRVL(x,enc_5label,saveName)
    dt_5label = dtFitAndSave(x,yl5,"5label")
    enc_5label = enc
    
if 1:
    print("Floor4 训练")
    x= x0_1
    y =y0_1
    print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
    
    num_labels = 2 
    nSamples,nFeatures =  x.shape
    enc = OneHotEncoder()
    y= y.reshape(nSamples,-1)
    
    print("y.shape:",y .shape,"y.type:", type(y) )
    enc.fit(y)
    yOneHot=enc.transform(y).toarray()
    saveName = "hybrid2_KerasSimple3_likeResnet_floor4.h5"
    kerasModel3_Floor4 = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
    yKeras_Floor4=getKerasResnetRVL(x,enc,saveName)
    dt_Floor4 = dtFitAndSave(x,yl5,"Floor4")
    enc_floor4 = enc
    
if 1:
    print("Floor3 训练")
    x= x10_2
    y =y10_2
    print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
    
    num_labels = 2 
    nSamples,nFeatures =  x.shape
    enc = OneHotEncoder()
    y= y.reshape(nSamples,-1)
    
    print("y.shape:",y .shape,"y.type:", type(y) )
    enc.fit(y)
    yOneHot=enc.transform(y).toarray()
    saveName = "hybrid2_KerasSimple3_likeResnet_floor3.h5"
    kerasModel3_Floor3 = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
    yKeras_Floor3=getKerasResnetRVL(x,enc,saveName)
    dt_Floor3 = dtFitAndSave(x,yl5,"Floor3")
    enc_floor3 = enc
    
if 1:
    print("Floor2 训练")
    x= x210_3
    y =y210_3
    print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
    
    num_labels = 2 
    nSamples,nFeatures =  x.shape
    enc = OneHotEncoder()
    y= y.reshape(nSamples,-1)
    
    print("y.shape:",y .shape,"y.type:", type(y) )
    enc.fit(y)
    yOneHot=enc.transform(y).toarray()
    saveName = "hybrid2_KerasSimple3_likeResnet_floor2.h5"
    kerasModel3_Floor2 = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
    yKeras_Floor2=getKerasResnetRVL(x,enc,saveName)
    dt_Floor2 = dtFitAndSave(x,yl5,"Floor2")
    enc_floor2 = enc
    
if 1:
    print("Floor1 训练")
    x= x3210_4
    y =y3210_4
    print("x.shape:",x .shape,"y.shape:",y .shape,"y.type:", type(y) )
    
    num_labels = 2 
    nSamples,nFeatures =  x.shape
    enc = OneHotEncoder()
    y= y.reshape(nSamples,-1)
    
    print("y.shape:",y .shape,"y.type:", type(y) )
    enc.fit(y)
    yOneHot=enc.transform(y).toarray()
    saveName = "hybrid2_KerasSimple3_likeResnet_floor1.h5"
    kerasModel3_Floor1 = kerasFitAndSaveSimple3LikeResnet(x,yOneHot,num_labels,saveName)     
    yKeras_Floor1=getKerasResnetRVL(x,enc,saveName)
    dt_Floor1 = dtFitAndSave(x,yl5,"Floor1")
    enc_floor1 = enc