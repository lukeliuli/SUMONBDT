#代码复制来自于mainTestCSVMLP3(hmcnf_keras)
########################################融合决策树和多层神经网络###########################################################

#######################################第一步读取数据
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

file1 = "./trainData/dataAllSim1000.csv"
print("reading data")
xyDataTmp = pd.read_csv(file1)
xyData = np.array(xyDataTmp)
nSamples, nDims= xyData.shape
x = xyData[:,0:22]
y = xyData[:,22:26]
ylabel = y
y1Level= y[:,0]#01
y2Level= y[:,1]#012
y3Level= y[:,2]#01234

print("x.shape:",x.shape,"y.shape:",y.shape)

del xyDataTmp #节省内存
del xyData #节省内存


#######################################融合决策树和多层神经网络###########################################################

#######################################第二步基于神经网络训练，这里采用简单神经网络，RESNET类似和HNCF三种方法进行训练
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
###简单模型1，没有隐藏层
def kerasFitAndSaveSimple1(x,yOneHot,num_labels):
    nSamples,features_size = x.shape
    build_model = tf.keras.Sequential()
    build_model.add(layers.Dense(num_labels, activation='sigmoid',name="layer1",input_shape=(features_size,)))
    #model = tf.keras.Model(inputs=[features], outputs=[build_model])
    
    #enc = OneHotEncoder()
    #enc.fit(y)  
    #yOnehot=enc.transform(y).toarray()
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    build_model.fit([x],[yOneHot],epochs=10000, batch_size=80000*1)
    build_model.save("kerasSimple1.h5")
    plot_model(build_model, to_file='KerasSimple1_noHiddenLayer.png', show_shapes=True)
    return build_model

###简单模型2，有隐藏层
def kerasFitAndSaveSimple2(x,yOneHot,num_labels):
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
    build_model.fit([x],[yOneHot],epochs=10000, batch_size=80000*1)
    build_model.save("kerasSimple2.h5")
    plot_model(build_model, to_file='KerasSimple2_HiddenLayer.png', show_shapes=True)
    return build_model


###开始训练
nSamples,features_size = x.shape
num_labels = 5
enc = OneHotEncoder()
y3Level = np.array(y3Level)
y3Level= y3Level.reshape(nSamples,-1)
print(y3Level)
enc.fit(y3Level)  
yOneHot=enc.transform(y3Level).toarray()
#simpleMode1 = kerasFitAndSaveSimple1(x,yOneHot,num_labels)
simpleMode2 = kerasFitAndSaveSimple2(x,yOneHot,num_labels)