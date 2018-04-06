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
            nn.Conv2d(3, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        if shallow:
            self.dnn = nn.Sequential(
                nn.Linear(16*5*5, 10),
            )
        else:
            self.dnn = nn.Sequential(
                nn.Linear(16*5*5, 64),
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

def get_data_mnist(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def to_var(x):
    x = torch.autograd.Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def train(batchsize=256, lr=0.01, epoch=20, model=None):
    if model == None:
        model = CNN(True)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data(batchsize)

    for epoch in range(epoch):
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
        print('epoch={}, loss={:.4f}'.format(epoch + 1, np.average(losses)))
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
    return model

def calculate_grad(model, loss):


    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    grad_norm = grad_all ** 0.5
    return grad_norm.data.numpy()


def analyze():
    model32 = train(batchsize=32)
    model64 = train(batchsize=64)
    model128 = train(batchsize=128)
    model1024 = train(batchsize=1024)
    model5120 = train(batchsize=5120)
    model10240 = train(batchsize=10240)
    model20480 = train(batchsize=20480)
    
    models = [(model32, 32), (model64, 64), (model128, 128), (model1024, 1024), (model5120, 5120), (model10240, 10240), (model20480, 20480)]
    #models = [(model20480, 20480)]
    train_loader, test_loader = get_data_mnist(256)
    loss_func = nn.CrossEntropyLoss()
    fout = open('hw1-3-3-2.csv', 'w')
    fout.write('batch_size,train_accuracy,train_loss,test_accuracy,test_loss,norm')
    for model, bsize in models:
        print('bsize=',bsize)
        correct = 0
        total = 0
        losses = []
        norm = []
        for data in test_loader:
            images, labels = data
            outputs = model(Variable(images))
            loss = loss_func(outputs, Variable(labels))
            losses.append(loss.data[0])
            _, predicted = torch.max(outputs.data, 1)
            norm.append(calculate_grad(model, loss))
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_acc = 100 * correct / total
        test_loss = np.average(losses)
        correct = 0
        total = 0
        losses = []
        for data in train_loader:
            images, labels = data
            outputs = model(Variable(images))
            loss = loss_func(outputs, Variable(labels))
            losses.append(loss.data[0])
            _, predicted = torch.max(outputs.data, 1)
            norm.append(calculate_grad(model, loss))
            total += labels.size(0)
            correct += (predicted == labels).sum()
        norm = np.average(np.array(norm))
        train_acc = 100 * correct / total
        train_loss = np.average(losses)
        print('train_acc=',train_acc)
        print('train_loss=', train_loss)
        print('test_acc=', test_acc)
        print('test_loss=', test_loss)
        print(norm)
        fout.write(str(bsize)+','+str(train_acc)+','+str(train_loss)+','+str(test_acc)+','+str(test_loss)+','+str(norm)+'\n')


def main():
    analyze()
    '''
    model = CNN(True)
    print(model)
    model.load_state_dict(torch.load('model/hw1-3-3-2/bsize32.pt'))

    model = train(batchsize=32, epoch=100, model=model)
    torch.save(model.state_dict(), 'model/hw1-3-3-2/bsize32.pt')
    '''


if __name__ == "__main__":
    main()
