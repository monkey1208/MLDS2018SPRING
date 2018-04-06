import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import count
from torchvision import datasets, transforms
import random

class MnistDNN(torch.nn.Module):
    def __init__(self, shallow):
        super(MnistDNN, self).__init__()

        if shallow:
            self.dnn = nn.Sequential(
                nn.Linear(784, 200),
                nn.ReLU(),
                nn.Linear(200, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
                nn.ReLU()
            )
        else:
            self.dnn = nn.Sequential(
                nn.Linear(16*4*4, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 10)
            )

    def forward(self, x):
        #x = self.cnn(x)
        #x = x.view(x.size(0), -1)
        out = self.dnn(x)
        return out

class MnistCNN(torch.nn.Module):
    def __init__(self, shallow):
        super(MnistCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        if shallow:
            self.dnn = nn.Sequential(
                nn.Linear(64*4*4, 128),
                nn.Linear(128, 10)
            )
        else:
            self.dnn = nn.Sequential(
                nn.Linear(16*4*4, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 10)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.dnn(x)
        return out

class CifarCNN(torch.nn.Module):
    def __init__(self, shallow):
        super(CifarCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        if shallow:
            self.dnn = nn.Sequential(
                nn.Linear(64*5*5, 80),
                nn.ReLU(),
                nn.Linear(80, 10)
            )
        else:
            self.dnn = nn.Sequential(
                nn.Linear(64*5*5, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.dnn(x)
        return out


def get_m_data(batch_size):
    train_set = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    #print(train_set.train_data)
    random.shuffle(train_set.train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_data(batch_size):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    random.shuffle(trainset.train_labels)
    print('random label')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    #dataloader = DataLoader()

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader

def to_var(x):
    x = torch.autograd.Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def test_loss(model, loss_func, test_loader):
    lss = 0
    total = 0
    for data in test_loader:
        images, labels = data
        labels = Variable(labels)
        outputs = model(Variable(images.view(images.size()[0], -1)))
        loss = loss_func(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        lss += loss.data[0]
    return lss / total

def main():
    fw = open('1-3-1.csv', 'w')
    fw.write('epoch,train_loss,test_loss\n')
    model = MnistDNN(True)
    print(model)
    #optimizer = optim.Adamax(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = get_m_data(256)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params:', params)
    for epoch in range(5000):
        losses = []
        for sample in train_loader:
            import ipdb;
            #ipdb.set_trace()
            x = to_var(sample[0].view(sample[0].size()[0], -1))
            y = to_var(sample[1])
            prediction = model(x)
            loss = loss_func(prediction, y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            losses.append(loss.data[0])
        print('epoch={}, loss={:.4f}'.format(epoch+1, np.average(losses)))

        t_loss = test_loss(model, loss_func, test_loader)
        print('test_loss={:.4f}'.format(t_loss))
        output_str = str(epoch+1)+','+str(np.average((losses)))+','+str(t_loss)+'\n'
        print(output_str)
        fw.write(output_str)
    fw.close()

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = model(Variable(images.view(images.size()[0], -1)))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))




if __name__ == "__main__":
    main()
