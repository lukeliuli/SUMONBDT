#!/usr/bin/env python
# coding: utf-8

# In[1]:


#mkdir /content/tmp
#%cp -r -f -v /content/drive/MyDrive/SUMONBDT /content/tmp
#from google.colab import drive
#drive.mount('/content/drive')
#%cd /content/drive/MyDrive/SUMONBDT
#get_ipython().magic('cd /home/liuli/github/SUMONBDT')
#!nvidia-smi
#用于测试oneHot
#############################################################也是第一步，读取数据
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



enc = OneHotEncoder()
#[2,3,5,9]
x1 = [0,0,0,0]
x2 = [0,0,0,1]

x3 = [1,1,1,2]
x4 = [1,1,1,3]
x5 = [1,1,2,4]
x6 = [1,1,2,5]
x7 = [1,2,3,6]
x8 = [1,2,3,7]
x9 = [1,2,4,8]
X = [x1, x2, x3,x4,x5,x6,x7,x8,x9]
enc.fit(X)
#print(enc.transform(X).toarray())


########################读写CSV,并转为oneHot
file1 = "./trainData/dataAllSim10000.csv"
print("reading data")
xyDataTmp = pd.read_csv(file1)
#print(xyDataTmp.info())
xyData = np.array(xyDataTmp)

x = xyData[:,0:22]
y = xyData[:,22:26]

y = enc.transform(y).toarray()

print("x.shape:",x.shape,"yOneHot.shape:",y.shape)



del xyDataTmp #节省内存
del xyData #节省内存



# In[ ]:



################################################################第二步，训练
#1. 核心为keras220不是pytorch
#2. 基于hmcnf
import model_hmcnf
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#hierarchy = [18, 80, 178, 142, 77, 4]
hierarchy = [2,3,5,9]
features_size = x.shape[1]
label_size = y.shape[1]
beta = 0.2
dropout_rate=0.1
relu_size=384



def local_model(num_labels, dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_labels, activation='sigmoid'))
    return model


def global_model(dropout_rate, relu_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(relu_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    return model


def sigmoid_model(label_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(label_size, activation='sigmoid',name="global"))
    return model

features = layers.Input(shape=(features_size,))
global_models = []
local_models = []


for i in range(len(hierarchy)):
    if i == 0:
        global_models.append(global_model(dropout_rate, relu_size)(features))
    else:
        global_models.append(global_model(dropout_rate, relu_size)(layers.concatenate([global_models[i-1], features])))

p_glob = sigmoid_model(label_size)(global_models[-1])


#显示只有全局模型的情况
#modelTmp1 = tf.keras.Model(inputs=[features], outputs=[p_glob])
#modelTmp1.summary()#
#plot_model(modelTmp1, to_file='Flatten1.png', show_shapes=True)


for i in range(len(hierarchy)):
    local_models.append(local_model(hierarchy[i], dropout_rate, relu_size)(global_models[i]))
    
#显示只有局部局模型的情况(部分全局)
p_loc = layers.concatenate(local_models)
#modelTmp2 = tf.keras.Model(inputs=[features], outputs=[p_loc])
#modelTmp2.summary()#
#plot_model(modelTmp2, to_file='Flatten2.png', show_shapes=True)
p_glob1 = layers.Lambda(lambda x: x*beta,name="global")(p_glob)
p_loc1 = layers.Lambda(lambda x: x*(1-beta),name="local")(p_loc)

labels = layers.add([p_glob1, p_loc1])

model = tf.keras.Model(inputs=[features], outputs=[labels])




plot_model(model, to_file='FlattenAll.png', show_shapes=True)


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['mae'])
model.fit([x],[y],epochs=1000, batch_size=25600*1)
model.save("hmcnf10000.h5")


# In[ ]:


##################################################################第三步，验证
get_ipython().magic('cd /content/drive/MyDrive/SUMONBDT')
import model_hmcnf
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import numpy as np
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#######################0.准备onehot
enc = OneHotEncoder()
#[2,3,5,9]
x1 = [0,0,0,0]
x2 = [0,0,0,1]

x3 = [1,1,1,2]
x4 = [1,1,1,3]
x5 = [1,1,2,4]
x6 = [1,1,2,5]
x7 = [1,2,3,6]
x8 = [1,2,3,7]
x9 = [1,2,4,8]
X = [x1, x2, x3,x4,x5,x6,x7,x8,x9]
enc.fit(X)

#######################2.准备数据
        
file1 = "./trainData/dataAllSim10000.csv"
print("reading data")
xyDataTmp = pd.read_csv(file1)
#print(xyDataTmp.info())
xyData = np.array(xyDataTmp)

x = xyData[:,0:22]
y = xyData[:,22:26]
ylabel = y
y = enc.transform(y).toarray()


del xyDataTmp #节省内存
del xyData #节省内存
#######################3.预测模型
print("3.HMCNF预测模型")
hierarchy = [2,3,5,9]
features_size = x.shape[1]
label_size = y.shape[1]
beta = 0.2

model_name ="hmcnf.h5" 

model = keras.models.load_model(model_name)
y_out = model.predict([x], batch_size=2560)
y_predict = np.where(y_out > 0.5, 1, 0)

predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)


print("validated {} , {} good out of {} samples".format(model_name, np.sum(predict_ok), predict_ok.shape[0]))
del y_predict #节省内存
del predict_ok #节省内存
#######################3.层次预测预测模型
print("3.层次预测预测模型")
y1 = np.where(y_out[:,0:2] > 0.5, 1, 0)
y2 = np.where(y_out[:,2:5] > 0.5, 1, 0)
y3 = np.where(y_out[:,5:10] > 0.5, 1, 0)
y4 = np.where(y_out[:,10:19] > 0.5, 1, 0)
for i in range(y4.shape[0]):
    tmp1 = y1[i]
    tmp2 = y2[i]
    tmp3 = y3[i]
    tmp4 = y4[i]
    if sum(tmp1) == 0:
        index=  np.argmax(tmp1)
        y1[i,index]=1
        
    if sum(tmp2) == 0:
        index=  np.argmax(tmp2)
        y2[i,index]=1
        
    if sum(tmp3) == 0:
        index=  np.argmax(tmp3)
        y3[i,index]=1
    
    if sum(tmp4) == 0:
        index=  np.argmax(tmp4)
        y4[i,index]=1
        #print(i,y4[i],index)
y_predict = np.concatenate([y1,y2,y3,y4],axis=1)
predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)
print("validated {} , {} good out of {} samples".format(model_name, np.sum(predict_ok), predict_ok.shape[0]))

#onehot 2 label
ypredict = enc.inverse_transform(y_predict)
del y_predict #节省内存
del predict_ok #节省内存
del y1,y2,y3,y4
#######################4.评估层次模型
#hierarchy = [2,3,5,9]

##第一层，2
print("###################################第一层，2")
h1_yp = ypredict[:,0]
h1_yl = ylabel[:,0]
tmp1 = classification_report(h1_yl,h1_yp)
tmp2 = confusion_matrix(h1_yl,h1_yp,normalize='true')
tmp3 = confusion_matrix(h1_yl,h1_yp,normalize='pred')
print(tmp1)
print(np.around(tmp2, decimals=3))
print(np.around(tmp3, decimals=3))


##第二层，3
print("################################第二层，3")
h2_yp = ypredict[:,1]
h2_yl = ylabel[:,1]
tmp1 = classification_report(h2_yl,h2_yp)
tmp2 = confusion_matrix(h2_yl,h2_yp,normalize='true')
tmp3 = confusion_matrix(h2_yl,h2_yp,normalize='pred')
print(tmp1)
print(np.around(tmp2, decimals=3))
print(np.around(tmp3, decimals=3))



##第三层，5
print("#############################第三层，5")
h3_yp = ypredict[:,2]
h3_yl = ylabel[:,2]
tmp1 = classification_report(h3_yl,h3_yp)
tmp2 = confusion_matrix(h3_yl,h3_yp,normalize='true')
tmp3 = confusion_matrix(h3_yl,h3_yp,normalize='pred')
print(tmp1)
print(np.around(tmp2, decimals=3))
print(np.around(tmp3, decimals=3))


##第四层，9
print("#############################第四层，9")
h4_yp = ypredict[:,3]
h4_yl = ylabel[:,3]
tmp1 = classification_report(h4_yl,h4_yp)
tmp2 = confusion_matrix(h4_yl,h4_yp,normalize='true')
tmp3 = confusion_matrix(h4_yl,h4_yp,normalize='pred')
print(tmp1)
print(np.around(tmp2, decimals=3))
print(np.around(tmp3, decimals=3))



# In[2]:


get_ipython().system('pwd')

