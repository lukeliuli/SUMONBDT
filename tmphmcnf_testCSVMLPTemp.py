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
###测试权重
nSamples =5000
input_dim = 10
#x = np.random.standard_normal(size=(nSamples, input_dim))
x = np.random.randint(low=0, high=10, size=(nSamples, input_dim))
y1 = np.zeros((nSamples, 1))#>50
y1A = np.zeros((nSamples, 1))#>50 and <60
y1B = np.zeros((nSamples, 1))#>=60
sumX = np.sum(x,axis=1)
index=np.where(sumX>40)
y1[index]=1
index=np.where((sumX>50)& (sumX<70))
y1A[index]=1
index=np.where(sumX>=70)
y1B[index]=1

##数据来源2
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
y1= y[:,0]



#测试决策树
def dtFitAndSave(x,y,class_names1,saveName):
    dt = tree.DecisionTreeClassifier(max_depth=5,min_samples_split=100,min_samples_leaf=100,min_impurity_split=0.06,ccp_alpha=0.001)
    dt = dt.fit(x, y)
    tree.plot_tree(dt)
    data=tree.export_graphviz(dt, out_file=None,class_names=class_names1,filled=True) 
    graph = graphviz.Source(data)
    graph.render(saveName)
    
    yPredict = dt.predict(x)
    tmp1 = classification_report(y,yPredict)
    print(tmp1)
    yPredict = dt.predict_proba(x)
    index = np.where((yPredict[:,1]<0.98)&(yPredict[:,1]>0.5))
    print(index[0].shape,index)
    index = np.where((yPredict[:,1]<0.90)&(yPredict[:,1]>0.5))
    print(index[0].shape,index)
    index = np.where((yPredict[:,1]<0.80)&(yPredict[:,1]>0.5))
    print(index[0].shape,index)
    index = np.where((yPredict[:,1]<0.70)&(yPredict[:,1]>0.5))
    print(index[0].shape,index)
    
    
    d_path = dt.decision_path(x[0])
    print(d_path)
    
    
    #print(yPredict[index,1])
    #plt.hist(yPredict[index,1],10,density=True)
    #plt.show()

dtFitAndSave(x,y1,["0","1"],"bigger")

###################################################################################
#测试神经网络
def kerasFitAndSave(x,y,num_labels):
    nSamples,features_size = x.shape
    relu_size = 384
    dropout_rate =0.1
    models=[]
    
    build_model = tf.keras.Sequential()
   
    build_model.add(layers.Dense(relu_size, activation='relu',name="layer1",input_shape=(features_size,)))
    build_model.add(layers.Dropout(dropout_rate,name="Dropout1-2"))
    build_model.add(layers.Dense(num_labels, activation='sigmoid',name="layer2"))
    
    #model = tf.keras.Model(inputs=[features], outputs=[build_model])
    plot_model(build_model, to_file='AKeras.png', show_shapes=True)
    
    enc = OneHotEncoder()
    enc.fit(y)  
    yOnehot=enc.transform(y).toarray()
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    build_model.fit([x],[yOnehot],epochs=100, batch_size=80000*1)
    build_model.save("Akeras.h5")
    plot_model(build_model, to_file='AKeras.png', show_shapes=True)
    
    return build_model,models

def kerasFitAndSaveSimple(x,y,num_labels):
    nSamples,features_size = x.shape
    relu_size = 2
    models=[]
    
    build_model = tf.keras.Sequential()
   
    build_model.add(layers.Dense(num_labels, activation='sigmoid',name="layer1",input_shape=(features_size,)))
    
    #model = tf.keras.Model(inputs=[features], outputs=[build_model])
    plot_model(build_model, to_file='AKerasSimple.png', show_shapes=True)
    
    enc = OneHotEncoder()
    enc.fit(y)  
    yOnehot=enc.transform(y).toarray()
    build_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    build_model.fit([x],[yOnehot],epochs=10000, batch_size=80000*1)
    build_model.save("Akeras.h5")
    plot_model(build_model, to_file='AKeras.png', show_shapes=True)
    
    return build_model,models

y1 = np.array(y1)
y1= y1.reshape(nSamples,-1)
#kerasFitAndSave(x,y1,2)
#kerasFitAndSaveSimple(x,y1,2)