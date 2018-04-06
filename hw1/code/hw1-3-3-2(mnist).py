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
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        if shallow:
            self.dnn = nn.Sequential(
                nn.Linear(16*4*4, 10)
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

def get_data(batch_size):
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

def train(batchsize=256, lr=0.01):
    model = CNN(True)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data(batchsize)

    for epoch in range(40):
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


def main():

    model64 = train(64)
    #model1024 = train(1024)
    #torch.save(model64.state_dict(), 'model/hw1-3-3-2-64.pt')
    #torch.save(model1024.state_dict(), 'model/hw1-3-3-2-1024.pt')
    torch.save(model64.state_dict(), 'model/mnist/hw1-3-3-2-64.pt')
    #torch.save(model1024.state_dict(), 'model/mnist/hw1-3-3-2-1024.pt')

    '''
    model64 = CNN(True)
    model1024 = CNN(True)
    model64.load_state_dict(torch.load('model/hw1-3-3-2-01.pt'))
    model1024.load_state_dict(torch.load('model/hw1-3-3-2-0001.pt'))
    cnn_weight_64_0 = model64.cnn[0].weight.data.numpy()
    cnn_weight_1024_0 = model1024.cnn[0].weight.data.numpy()
    cnn_weight_64_1 = model64.cnn[3].weight.data.numpy()
    cnn_weight_1024_1 = model1024.cnn[3].weight.data.numpy()
    dnn_weight_64 = model64.dnn[0].weight.data.numpy()
    dnn_weight_1024 = model1024.dnn[0].weight.data.numpy()
    loss_func = nn.CrossEntropyLoss()
    alphas = []
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []
    for a in range(-20, 41):
        alpha = a / 20
        alphas.append(alpha)
        print(alpha)
        model = CNN(True)
        model.cnn[0].weight.data = torch.from_numpy((1 - alpha) * cnn_weight_64_0 + alpha * cnn_weight_1024_0)
        model.cnn[3].weight.data = torch.from_numpy((1 - alpha) * cnn_weight_64_1 + alpha * cnn_weight_1024_1)
        model.dnn[0].weight.data = torch.from_numpy((1 - alpha) * dnn_weight_64 + alpha * dnn_weight_1024)
        #model1024.cnn[0].weight.data = torch.from_numpy(np.zeros(model1024.cnn[0].weight.data.numpy().shape))
        #print(model1024.cnn[0].weight.data.numpy())

        train_loader, test_loader = get_data(256)
        correct = 0
        total = 0
        losses = []
        for data in test_loader:
            images, labels = data
            outputs = model(Variable(images))
            loss = loss_func(outputs, Variable(labels))
            losses.append(loss.data[0])
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        test_accuracy.append((100 * correct / total))
        test_losses.append(np.average(losses))
        correct = 0
        total = 0
        losses = []
        for data in train_loader:
            images, labels = data
            outputs = model(Variable(images))
            loss = loss_func(outputs, Variable(labels))
            losses.append(loss.data[0])
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        train_accuracy.append((100 * correct / total))
        train_losses.append(np.average(losses))
    fout = open('1-3-3-1-lr.csv', 'w')
    fout.write('alpha,train_accuracy,train_loss,test_accuracy,test_lost\n')
    for i in range(len(alphas)):
        fout.write(str(alphas[i])+','+str(train_accuracy[i])+','+
                   str(train_losses[i])+','+str(test_accuracy[i])+','+
                   str(test_losses[i])+'\n')
    fout.close()
    '''

if __name__ == "__main__":
    main()
