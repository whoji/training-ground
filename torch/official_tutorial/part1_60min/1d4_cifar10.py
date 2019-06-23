# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as pyplot
import numpy as np
import torch.optim as opt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("we are doing this on: "+str(device) + " !!!")

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

def not_runable():
    def imshow(img):
        img = img / 2 + 0.5 # unnormalize
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1,2,0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)  # in 3 ; out 6 ; 5x5
        self.pool = nn.MaxPool2d(2,2)
        self.Conv2d = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5* 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
LOSS = nn.CrossEntropyLoss()
opt = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)


## train the network

# net.to(device)

for epoch in range(2):
    running_loss =0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # inputs, labels = data.to(device)
        opt.zero_grad()
        outputs = net(inputs)
        loss = LOSS(outputs, inputs)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % 2000 == 1999: # this is so strange
            print("[%d, %5d] loss %.3f" % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print("GGWP")

# test the test dataset

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# test on whole dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# accuracy by class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# question to clarify
# 1. how to run these iterator and enumerator
# 2. how to run on gpu
# 3. torch.max dim