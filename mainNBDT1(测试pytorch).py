import os
import resnet
from collections import namedtuple
import time
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch
%cd / content/drive/MyDrive/SUMONBDT


outputNow = torch.Tensor()
inputNow = torch.Tensor()
fc_weight = torch.Tensor()
fc_bias = torch.Tensor()


def hook2(module, input, output):
    '''把这层的输出拷贝到features中'''
    global inputNow
    global outputNow
    global fc_weight
    global fc_bias

    #print(input[0].shape)
    inputNow = input[0].clone()
    outputNow = output.clone()
    fc_weight = module.weight.clone()
    fc_bias = module.bias.clone()
    #inputNow =  input

    #print(output.data)
    #print(input[0].data)
    #print(input[0].data.size())
    #print(module.weight)
    #print(module.bias)
    #print(inputNow)
    #print(outputNow)

    #computeLoss()


def computeLoss(labels):

  global inputNow
  global outputNow
  global fc_weight
  global fc_bias
  global myLoss

  ###简单的分级识别
  #GOD: 1. plane,bird
  #     2. deer,dog,horse,cat,frog
  #     3. car,truck,ship
  #classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
  #           'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7 , 'ship' 8,  'truck' 9)
  #layerA         0
  #layerB      0 (likebird)       1 (likecat)         2(likecar)
  #layerC  plane 0,bird2     deer4,dog5,horse7,cat3,frog6        car 1,truck9,ship8
  myDict = torch.LongTensor([0, 2, 0, 1, 1, 1, 1, 1, 2, 2])
  #print(myDict)
  w0 = fc_weight  # 10x512
  bias = fc_bias  # 10x512
  wA_B = {}
  bA_B = {}

  wA_B['0_1_likebird'] = (w0[0]+w0[2])/2
  bA_B['0_1_likebird'] = (bias[0]+bias[2])/2

  wA_B['0_2_likecat'] = (w0[3]+w0[4]+w0[5]+w0[6]+w0[7])/6
  bA_B['0_2_likecat'] = (bias[3]+bias[4]+bias[5]+bias[6]+bias[7])/6

  wA_B['0_3_likecar'] = (w0[1]+w0[8]+w0[9])/3
  bA_B['0_3_likecar'] = (bias[1]+bias[8]+bias[9])/3
  wTmp = wA_B['0_1_likebird'].unsqueeze(1)
  s1 = torch.mm(inputNow, wTmp)+bA_B['0_1_likebird']
  wTmp = wA_B['0_2_likecat'].unsqueeze(1)
  s2 = torch.mm(inputNow, wTmp)+bA_B['0_2_likecat']
  wTmp = wA_B['0_3_likecar'].unsqueeze(1)
  s3 = torch.mm(inputNow, wTmp)+bA_B['0_3_likecar']

  s4 = torch.cat((s1, s2, s3), 1)

  #_, predicted = s4.max(1)

  #print("s1:",s1)
  #print("s2:",s2)
  #print("s3:",s3)
  #print("s4:",s4)
  #print("myPredicted:",predicted)

  myLabels = labels.clone().detach()
  #print("mylabels1:",myLabels)
  for i, data in enumerate(labels):
    #tmp = data.numpy()
    #print(data)
    myLabels[i] = myDict[data]
  #print("mylabels2:",myLabels)

  myLoss = criterion(s4, myLabels)
  #print("myLoss:",myLoss)
  return myLoss, myLabels


#########################################################################################
transform1 = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差


#transform1 = transforms.Compose(
#    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差

batch_sizeV = 512
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform1)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_sizeV, shuffle=True, num_workers=2)

batch_sizeV = 512
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_sizeV, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

startEpoch = 0

#resnet18 = models.resnet18(pretrained=False)#采用torchvision的模型，无法达到94%的正确率，最多88%
resnet18 = resnet.resnet18(num_classes=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device ="cpu"
if device == 'cuda':
  resnet18 = torch.nn.DataParallel(resnet18)
  cudnn.benchmark = True
print(device)
resnet18 = resnet18.to(device)


modelPathName = "/content/drive/MyDrive/SUMONBDT/trainedModes/resnet18End_accuray95.modeparams"
modelPath = "/content/drive/MyDrive/SUMONBDT/trainedModes/"
params = torch.load(modelPathName, map_location=device)
resnet18.load_state_dict(params["net"])
startEpoch = params["epoch"]
#print(params)


#######################################################################
##基于hook和name_parameters.看输出数据类型和值,网络权值
for name, parameters in resnet18.named_parameters():
    #print(name, ':', parameters.size())
    params[name] = parameters.detach()
#print(params["fc.bias"])
handle = resnet18.fc.register_forward_hook(hook2)
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)
outputs = resnet18(images)
_, predicted = torch.max(outputs.data, 1)

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
myLoss, myLabels = computeLoss(labels)
#print(outputs.data.numpy())
print("inputNow:", inputNow)
print("outputNow:", outputNow)
print("predicted:", predicted)
print("labels:", labels)
print("myLabels:", myLabels)
print("loss:", loss)
print("myLoss:", myLoss)

handle.remove()

##########################################################################################
###开始训练
print("##开始训练################################################################################")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device ="cpu"
if device == 'cuda':
  resnet18 = torch.nn.DataParallel(resnet18)
  cudnn.benchmark = True
print(device)
resnet18 = resnet18.to(device)

# trainloader
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)

startEpoc = 0
epochs = 1000
resnet18.train()
print("start training")

handle = resnet18.fc.register_forward_hook(hook2)

for epoch in range(startEpoch, epochs):  # 多批次循环

    resnet18.train()
    running_loss = 0.0
    time_start = time.time()
    #learning rate 不变
#     #adjust_learning_rate(optimizer, epoch, epochs, trainloader, batch_sizeV)
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
         # 获取输入
         #print(epoch,i)
         inputs, labels = data

         inputs, labels = inputs.to(device), labels.to(device)

         # 梯度置0
         optimizer.zero_grad()

         # 正向传播，反向传播，优化a
         outputs = resnet18(inputs)

         loss = criterion(outputs, labels)

         ###############
         myLoss, myLabels = computeLoss(labels)
         loss = loss+0.1*myLoss
         loss.backward()
         optimizer.step()

         _, predicted = outputs.max(1)
         total += labels.size(0)
         correct += predicted.eq(labels).sum().item()
         #print("myLoss:",myLoss)
         #print("myLabels:",myLabels)
         #print("Loss:",loss)
         #print("labels:",labels)
         #print("i,epoch:",i,epoch)
         # 打印状态信息
         running_loss += loss.item()
         if i % 20 == 19:  # 每200批次打印一次
             time_end = time.time()
             print('one 20 batch totally time cost %.3f' %
                   (time_end-time_start))
             print("batchIndex %d |trainLen %d | Loss: %.3f | Acc: %.3f | correct, total: (%d,%d)" % (
                 i, len(trainloader), running_loss/(i+1), 100.*correct/total, correct, total))

     time_end = time.time()
     print('epoch %d totally time cost %.3f' % (epoch, time_end-time_start))
     state = {"net": resnet18.state_dict(
     ), "optimizer": optimizer.state_dict(), "epoch": epoch}
     modelPath = "/content/drive/MyDrive/SUMONBDT/trainedModes"
     modelPathNameTmp = modelPath+"/nbdt_resnet18_"+str(epoch)+".modeparams"
     torch.save(state, modelPathNameTmp)
     params = torch.load(modelPathNameTmp)
     resnet18.load_state_dict(params["net"])
     optimizer.load_state_dict(params["optimizer"])
     #evalCifar(testloader, resnet18, classes)
handle.remove()
print('Finished Training')

#有两种方法，一种只保存参数，一种全保存，后者简单但存储量大，我用的是前者
# # 保存和加载整个模型
state = {"net": resnet18.state_dict(
), "optimizer": optimizer.state_dict(), "epoch": epoch}
modelPath = "/content/drive/MyDrive/SUMONBDT/trainedModes"
modelPathName = modelPath+"/nbdt_resnet18_End"+".modeparams"
torch.save(state, modelPathName)
params = torch.load(modelPathName)
resnet18.load_state_dict(params['net'])
optimizer.load_state_dict(params['optimizer'])

# # 仅保存和加载模型参数(推荐使用)
#torch.save(resnet18.state_dict(), './trainedModes/resnet18params.pkl')
#resnet18.load_state_dict(torch.load('./trainedModes/resnet18params.pkl'))
