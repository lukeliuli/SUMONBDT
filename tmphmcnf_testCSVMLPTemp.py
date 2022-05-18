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
        
file1 = "./trainData/dataAllSim.csv"
print("reading data")
xyDataTmp = pd.read_csv(file1)
#print(xyDataTmp.info())
xyData = np.array(xyDataTmp)

x = xyData[:,0:22]
y = xyData[:,22:26]
ylabel = y
y = enc.transform(y).toarray()



#######################3.预测模型

hierarchy = [2,3,5,9]
features_size = x.shape[1]
label_size = y.shape[1]
beta = 0.5

model_name ="hmcnf.h5" 

model = keras.models.load_model(model_name)
y_out = model.predict([x], batch_size=2560)
y_predict = np.where(y_out > 0.5, 1, 0)

predict_ok = np.where(np.sum(y_predict - y, axis=1) == 0, 1, 0)


print("validated {} , {} good out of {} samples".format(model_name, np.sum(predict_ok), predict_ok.shape[0]))
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

#######################4.评估层次模型
#hierarchy = [2,3,5,9]
ypredict = enc.inverse_transform(y_predict)

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

