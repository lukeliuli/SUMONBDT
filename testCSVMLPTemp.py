###用于测试csv和mlp
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import time
from collections import namedtuple

from torch.utils.data.sampler import WeightedRandomSampler

import torch.nn as nn
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd


print(torch.__version__)  #注意是双下划线
input()
# 五层神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

mlp = NeuralNet(22,200,4)
print(mlp)

##############################################读写CSV
file1 = "./trainData/dataAllSim.csv"
print("reading data")
xyDataTmp = pd.read_csv(file1)
print(xyDataTmp.info())
xyData = np.array(xyDataTmp)


##方式1
#data1=[]
#csv_reader = csv.reader(open(file1))
#for row in csv_reader:
#        data1.append(row)
#print(data1[0])        

##方式2
#https://blog.csdn.net/zw__chen/article/details/82806900
print("reading data")
#csvfile = open(file1, 'r')
#xyData= np.loadtxt(file1,delimiter=',',skiprows=1)
x = xyData[:,0:-1]
y = xyData[:,-1]
print(y.shape )
print(x.shape )
print("reading data,to Torch")
x1 =torch.from_numpy(x)
y1 =torch.from_numpy(y)
dataset1 = TensorDataset(x1.float(),y1.long())


#train_dataset, test_dataset = random_split(
#    dataset=dataset1,
#    lengths=[7, 3],
#    generator=torch.Generator().manual_seed(0)
#)

#################################
#数据准备：平衡
#tmp = y
#trainratio = np.bincount(tmp.astype("int64"))
#train_weights = 1./trainratio
#train_weights[0]= train_weights[0]*2
#train_sampleweights = torch.from_numpy(train_weights)
#train_sampler = WeightedRandomSampler(weights=train_sampleweights, num_samples = len(y))
#train_loader = DataLoader( dataset1, batch_size=5120, num_workers=1, sampler=train_sampler)

print("loading data")
train_loader = DataLoader(dataset=dataset1, batch_size=2048, shuffle=True)

#准备设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    mlp = torch.nn.DataParallel(mlp)
    cudnn.benchmark = True
    mlp =  mlp.to(device)
print(device)

#准备模型
print("loading net")
modelPathName = "./trainedModes/"+"redTLS.modeparams"
params = torch.load(modelPathName)
mlp.load_state_dict(params['net'])


#################################
#训练开始
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
running_loss = 0
startEpoch = 0
epochs = 3000#大概8小时
print("start training")

for epoch in range(startEpoch, epochs):  # 多批次循环

    mlp.train()
    time_start = time.time()
    total = 0
    correct = 0
    running_loss = 0.0
    Alllabels = []
    Allpredicted = []
    for i, data in enumerate(train_loader, 0):
        # 获取输入
        #print(epoch,i)
        inputs, labels = data
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs=inputs.type(torch.float32)
        labels=labels

        if device == 'cuda':
            inputs, labels = inputs.to(device), labels.to(device)

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化a
        outputs = mlp(inputs)
        #print("inputs:",inputs)
        #print("###################################")
        #print("outputs:",outputs)
        #print("labels",labels)

        loss = criterion(outputs, labels)
        #print("loss:",loss.item())

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        Alllabels.extend(labels.numpy())
        Allpredicted.extend(predicted.numpy())
        #print("labels.size(0):",labels.size(0))
        #print("predicted:",predicted)
        #print("predicted.eq(labels).sum().item():",predicted.eq(labels).sum().item())
        
        running_loss += loss.item()

        #os.input()
        if i % 100 == 99:  # 每100批次打印一次
         time_end = time.time()
         print('##########epoch：%d,one 100 batch totally time cost %.3f' %(epoch,time_end-time_start))
         print("batchIndex %d |trainLen %d | Loss: %.3f | Acc: %.3f | correct, total: (%d,%d)" % (
             i, len(train_loader), running_loss/(i+1), 100.*correct/total, correct, total))

    time_end = time.time()
    
    print("end %d epoch,totally time cost %.3f" %(epoch,time_end-time_start))
    state = {"net": mlp.state_dict(), "optimizer": optimizer.state_dict()}
    modelPathName = "./trainedModes/"+"redTLS.modeparams"
    torch.save(state, modelPathName)
    params = torch.load(modelPathName)
    mlp.load_state_dict(params['net'])
    optimizer.load_state_dict(params['optimizer'])
    
    
    #用全训练集合进行评估
    #print("start eavl")
    #mlp.eval()
    #labels = y1.long()
    #print("start eavl 1")
    #outputs = mlp(x1.type(torch.float32))
    #print("start eavl 2")
    #_, predicted = torch.max(outputs.data,1)
    
    #print("start eavl 3")
    #total += labels.size(0)
    #correct += predicted.eq(labels).sum().item()
    
    #print("\n\n Eval--epochIndex %d |time: %d | Acc: %.3f | correct, total: (%d,%d)" % (
    #     epoch, (time_end-time_start), 100.*correct/total, correct, total))
    tmp1 = classification_report(Alllabels,Allpredicted)*100
    tmp2 = confusion_matrix(Alllabels,Allpredicted,normalize='true')
    tmp3 = confusion_matrix(Alllabels,Allpredicted,normalize='pred')
    print(tmp1)
    print(np.around(tmp2, decimals=3))
    print(np.around(tmp3, decimals=3))
  
    print("########################################################################")
