#在google中运行，https://colab.research.google.com/
from google.colab import drive
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
import torch.backends.cudnn as cudnn
import resnet



features = t.Tensor()
def hook(module, input, output):
    '''把这层的输出拷贝到features中'''
    features.copy_(output.data)
    print(output.data)



def evalCifar(testloader, resnet18, classes):

    correct = 0
    total = 0
    resnet18.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         labels = labels.to(device)
    #         images = images.to(device)
    #         outputs = resnet18(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1

    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


#第一步用RESNET 训练CIFAR10
#https: // github.com/lukeliuli/pytorch-handbook/blob/master/chapter1/4_cifar10_tutorial.ipynb
#https: // github.com/fengdu78/Data-Science-Notes/tree/master/8.deep-learning/PyTorch_beginner
#https: // blog.csdn.net/wudibaba21/article/details/106495118/




transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差

transform1 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # R,G,B每层的归一化用到的均值和方差

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sizeV,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sizeV,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


startEpoch = 0
#resnet18 = models.resnet18(pretrained=False)#采用torchvision的模型，无法达到94%的正确率，最多88%
resnet18 = resnet.resnet18(num_classes=10)
modelPathName = "./trainedModes/resnet18End_accuray95.modeparams"
params = torch.load(modelPathName)
resnet18.load_state_dict(params["net"])
startEpoch =params["epoch"]

################################

sub_model = resnet18.features
for name, module in sub_model._modules.items():
    x = module(x)
    print("名称:{}".format(name))

for name, parameters in resnet18.named_parameters():
    print(name, ':', parameters.size())
    parm[name] = parameters.detach().numpy()
#################################

handle = resnet18.layer8.register_forward_hook(hook)

evalCifar(testloader, resnet18, classes)

handle.remove()

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
