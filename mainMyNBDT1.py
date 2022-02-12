#第一，代码实现基本识别

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
import resnet

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差


#transform1 = transforms.Compose(
#    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, bwatch_size=batch_sizeV,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sizeV,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

startEpoch = 0

#GOD: 1. plane,bird
#     2. deer,dog,horse,cat,frog
#     3. car,truck,ship


startEpoch = 0
#resnet18 = models.resnet18(pretrained=False)#采用torchvision的模型，无法达到94%的正确率，最多88%
resnet18 = resnet.resnet18(num_classes=10)
#modelPathName="./resnet18End.modeparams"
#params = torch.load(modelPathName)
#resnet18.load_state_dict(params["net"])
#startEpoch =params["epoch"]


# 获取随机数据
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图像
imshow(torchvision.utils.make_grid(images))

#GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
resnet18 = resnet18.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
if device == 'cuda':
    resnet18 = torch.nn.DataParallel(resnet18)
    cudnn.benchmark = True

resnet18.train()
print("start training")
for epoch in range(startEpoch, epochs):  # 多批次循环

    running_loss = 0.0
    time_start = time.time()
    adjust_learning_rate(optimizer, epoch, epochs, trainloader, batch_sizeV)
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化a
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

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
    modelPathNameTmp = "./resnet18_"+str(epoch)+".modeparams"
    torch.save(state, modelPathNameTmp)
    params = torch.load(modelPathNameTmp)
    resnet18.load_state_dict(params["net"])
    optimizer.load_state_dict(params["optimizer"])
    evalCifar(testloader, resnet18, classes)
