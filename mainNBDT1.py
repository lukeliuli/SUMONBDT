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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


#第一步用RESNET 训练CIFAR10
#https: // github.com/lukeliuli/pytorch-handbook/blob/master/chapter1/4_cifar10_tutorial.ipynb
#https: // github.com/fengdu78/Data-Science-Notes/tree/master/8.deep-learning/PyTorch_beginner
#https: // blog.csdn.net/wudibaba21/article/details/106495118/


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

resnet18 = models.resnet18(pretrained=True)


# 获取随机数据
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图像
imshow(torchvision.utils.make_grid(images))

#GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
resnet18.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
print("start training")
for epoch in range(2):  # 多批次循环

    resnet18.train()
    running_loss = 0.0
    time_start = time.time()
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

        # 打印状态信息
        running_loss += loss.item()
        if i % 200 == 199:    # 每20批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            time_end = time.time()
            print('one 200 batch totally cost %.3f' % (time_end-time_start))
            running_loss = 0.0

    time_end = time.time()
    print('one epoch totally cost %.3f' % (time_end-time_start))

print('Finished Training')

#有两种方法，一种只保存参数，一种全保存，后者简单但存储量大，我用的是后者
# 保存和加载整个模型
torch.save(resnet18, './trainedModes/resnet18.pkl')
resnet18 = torch.load('./trainedModes/resnet18.pkl')

# 仅保存和加载模型参数(推荐使用)
#torch.save(resnet18.state_dict(), './trainedModes/resnet18params.pkl')
#resnet18.load_state_dict(torch.load('./trainedModes/resnet18params.pkl'))

images = images.to(device)
resnet18.eval()
outputs = resnet18(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


correct = 0
total = 0
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


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        labels = labels.to(device)
        images = images.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

'''
model = wrn28_10_cifar10(pretrained=True)
net = model(pretrained=True, num_classes=len(trainset.classes))
generate_dt(dataset='Imagenet1000',arch='wrn28_10_cifar10', model=model)
'''
