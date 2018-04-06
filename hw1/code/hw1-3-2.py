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

class CNN(torch.nn.Module):
    def __init__(self, shallow):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(1, 1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        if shallow:
            self.dnn = nn.Sequential(
                nn.Linear(16*5*5, 80),
                nn.ReLU(),
                nn.Linear(80, 10)
            )
        else:
            self.dnn = nn.Sequential(
                nn.Linear(5*5, 10),
                nn.ReLU(),
                #nn.Linear(64, 64),
                #nn.ReLU(),
                #nn.Linear(64, 64),
                #nn.ReLU(),
                #nn.Linear(64, 10)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.dnn(x)
        return out

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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader

def to_var(x):
    x = torch.autograd.Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def main():
    fw = open('hw1-3-2.csv', 'w')
    fw.write('params,accuracy\n')
    model = CNN(False)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data(256)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params:', params)
    for epoch in range(10):
        losses = []
        for sample in train_loader:
            x = to_var(sample[0])
            y = to_var(sample[1])
            prediction = model(x)
            loss = loss_func(prediction, y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            losses.append(loss.data[0])
        print('epoch={}, loss={:.4f}'.format(epoch+1, np.average(losses)))
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            outputs = model(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct/total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
    out_s = str(params)+','+str(accuracy)+'\n'
    fw.write(out_s)

if __name__ == "__main__":
    main()
