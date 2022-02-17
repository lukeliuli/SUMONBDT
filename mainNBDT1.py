#在google中运行，https://colab.research.google.com/

import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import time
from collections import namedtuple
#import resnet
import content.drive.MyDrive.SUMONBDT.resnet

inputNow = = torch.Tensor()
outputNow = = torch.Tensor()

def hook(module, input, output):
    '''把这层的输出拷贝到features中'''
    global inputNow
    global outputNow

    print(input[0].shape)
    inputNow = input[0].clone().detach()
    outputNow = output.clone().detach()
    #print(output.data)
    #print(input[0].data)
    #print(input[0].data.size())
    #print(module.weight)
    #print(module.bias)


transform1 = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差


#transform1 = transforms.Compose(
#    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差

batch_sizeV = 512
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_sizeV, shuffle=True, num_workers=2)

batch_sizeV = 1
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_sizeV, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

startEpoch = 0

#GOD: 1. plane,bird
#     2. deer,dog,horse,cat,frog
#     3. car,truck,ship


startEpoch = 0
#resnet18 = models.resnet18(pretrained=False)#采用torchvision的模型，无法达到94%的正确率，最多88%
resnet18 = resnet.resnet18(num_classes=10)
modelPathName = "/content/drive/MyDrive/SUMONBDT/trainedModes/resnet18End_accuray95.modeparams"
params = torch.load(modelPathName, map_location='cpu')
resnet18.load_state_dict(params["net"])
startEpoch = params["epoch"]
#print(params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
resnet18 = resnet18.to(device)

#######################################################################
##看输出数据类型和值
for name, parameters in resnet18.named_parameters():
    #print(name, ':', parameters.size())
    params[name] = parameters.detach()
print(params["fc.bias"])

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = resnet18(images)
_, predicted = torch.max(outputs.data, 1)
print(outputs.data.numpy())
print(predicted)
print(labels)

#######################################################################
##看中间层输出和值
handle = resnet18.fc.register_forward_hook(hook)
outputs = resnet18(images)
handle.remove()
print("inputNow:", inputNow)
print("outputNow:", outputNow)

w0 = params["fc.weight"]  # 10x512
bias = params["fc.bias"]  # 10x512

outTmp = torch.mm(inputNow, w0.transpose(0, 1))+bias
_, predictedTmp = torch.max(outTmp.data, 1)
print("outTmp:", outTmp)
print("predictedTmp:", predictedTmp)
########################################################################

########################################################################
###获得
#GOD: 1. plane,bird
#     2. deer,dog,horse,cat,frog
#     3. car,truck,ship
#classes = ('plane' 0, 'car' 1, 'bird' 2, 'cat' 3,
#           'deer' 4, 'dog' 5, 'frog' 6, 'horse' 7 , 'ship' 8,  'truck' 9)
#layerA         0 
#layerB      1 (likebird)       2 (likecat)                           3(likecar)
#layerC  plane,bird     deer,dog,horse,cat,frog        car,truck,ship

# w0 = params["fc.weight"].numpy()  # 10x512
# bias = params["fc.bias"].numpy()  # 10x512
# wA_B = {}
# bA_B = {}
# wA_B['0_1_likebird'] =  (w0[0]+w0[2])/2
# bA_B['0_1_likebird'] = (bias[0]+bias[2])/2


# wA_B['0_2_likecat'] = (w0[3]+w0[4]+w0[5]+w0[6]+w0[7])/6
# bA_B['0_2_likecat'] = (bias[3]+bias[4]+bias[5]+bias[6]+bias[7])/6

# wA_B['0_3_likecar'] = (w0[1]+w0[8]+w0[9])/3
# bA_B['0_3_likecar'] = (bias[1]+bias[8]+bias[9])/3


# s1 = np.dot(inputTmp, wA_B['0_1_likebird'])+bA_B['0_1_likebird']
# s2 = np.dot(inputTmp, wA_B['0_2_likecat'])+bA_B['0_2_likecat']
# s3 = np.dot(inputTmp, wA_B['0_3_likecar'])+bA_B['0_3_likecar']
# print(s1)
# print(s2)
# print(s3)

# z = np.dot(inputNow, w0.T)

###############################################################
w0 = params["fc.weight"]  # 10x512
bias = params["fc.bias"]  # 10x512

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

print(s1)
print(s2)
print(s3)
###############################################################################

# ##纯训练相关据
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # 展示图像
# imshow(torchvision.utils.make_grid(images))

# #GPU or CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# resnet18 = resnet18.to(device)

# trainloader
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(resnet18.parameters(), lr=0.01,
#                       momentum=0.9, weight_decay=5e-4)
# if device == 'cuda':
#     resnet18 = torch.nn.DataParallel(resnet18)
#     cudnn.benchmark = True


# resnet18.train()
# print("start training")
# for epoch in range(startEpoch, epochs):  # 多批次循环

#     resnet18.train()
#     running_loss = 0.0
#     time_start = time.time()
#     #learning rate 不变
#     #adjust_learning_rate(optimizer, epoch, epochs, trainloader, batch_sizeV)
#     total = 0
#     correct = 0
#     for i, data in enumerate(trainloader, 0):
#         # 获取输入
#         inputs, labels = data

#         inputs, labels = inputs.to(device), labels.to(device)

#         # 梯度置0
#         optimizer.zero_grad()

#         # 正向传播，反向传播，优化a
#         outputs = resnet18(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#         # 打印状态信息
#         running_loss += loss.item()
#         if i % 20 == 19:  # 每200批次打印一次
#             time_end = time.time()
#             print('one 20 batch totally time cost %.3f' %
#                   (time_end-time_start))
#             print("batchIndex %d |trainLen %d | Loss: %.3f | Acc: %.3f | correct, total: (%d,%d)" % (
#                 i, len(trainloader), running_loss/(i+1), 100.*correct/total, correct, total))

#     time_end = time.time()
#     print('epoch %d totally time cost %.3f' % (epoch, time_end-time_start))
#     state = {"net": resnet18.state_dict(
#     ), "optimizer": optimizer.state_dict(), "epoch": epoch}
#     modelPathNameTmp = "./resnet18_"+str(epoch)+".modeparams"
#     torch.save(state, modelPathNameTmp)
#     params = torch.load(modelPathNameTmp)
#     resnet18.load_state_dict(params["net"])
#     optimizer.load_state_dict(params["optimizer"])
#     evalCifar(testloader, resnet18, classes)

# print('Finished Training')

# #有两种方法，一种只保存参数，一种全保存，后者简单但存储量大，我用的是后者
# # 保存和加载整个模型

# torch.save(state, modelPathName)
# params = torch.load(modelPathName)
# resnet18.load_state_dict(params['net'])
# optimizer.load_state_dict(params['optimizer'])

# # 仅保存和加载模型参数(推荐使用)
# #torch.save(resnet18.state_dict(), './trainedModes/resnet18params.pkl')
# #resnet18.load_state_dict(torch.load('./trainedModes/resnet18params.pkl'))

# images = images.to(device)
# resnet18.eval()
# outputs = resnet18(images)
# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# ###
# evalCifar(testloader, resnet18, classes)
# '''
# model = wrn28_10_cifar10(pretrained=True)
# net = model(pretrained=True, num_classes=len(trainset.classes))
# generate_dt(dataset='Imagenet1000',arch='wrn28_10_cifar10', model=model)
# '''
