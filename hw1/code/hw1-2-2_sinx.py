import torch
import torch.utils.data as Data
from torch.autograd import Variable
from torch.autograd import grad
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
plt.switch_backend('agg')
class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_hid, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature,100)
        self.predict = nn.Linear(100,n_output)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.predict(x)
        return x

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 32


net = Net(n_feature=1, n_hidden=500, n_hid = 500, n_output=1)
print(net)

train_x = torch.unsqueeze(torch.linspace(-10,10,1000),dim = 1)
train_y = torch.sin(train_x)
train_x, train_y = Variable(train_x), Variable(train_y)
#plt.scatter(train_x.data.numpy(), train_y.data.numpy())
#optimizer = torch.optim.Adam(net.parameters(),lr = 0.1)
optimizer = torch.optim.LBFGS(net.parameters(), lr=0.005)
loss_func = nn.MSELoss()

gra = []
weight = []
plt.ion()
grad_zero = 1
losspoint = []
min_ratio = []
#hess = np.array([])
for t in range(100):
 #   for step, (batch_x, batch_y) in enumerate(loader):
    prediction = net(train_x)
    loss = loss_func(prediction, train_y)
    hessian = torch.autograd.grad(loss, net.parameters(), create_graph=True)[0]

    #for x in hessian:
    hess = (tuple((torch.autograd.grad(x, net.parameters(), retain_graph=True)[0]
                    for x in hessian)))
    #print(hess)
    '''hess = tuple(torch.stack([torch.autograd.grad(y_x[i], x, retain_graph=True)[0].data
                            for i in range(x.size(0))]))
                for x, y_x in zip(net.parameters(), hessian))'''
    hess = np.array(([x.data.numpy().flatten() for x in hess]))


    w, c = LA.eig(hess)
    #import ipdb; ipdb.set_trace()
    #hessian_x = torch.autograd.grad(hessian, net.parameters())

    #print(net.weight.grad)
    optimizer.zero_grad()
    loss.backward()
    def closure():
        optimizer.zero_grad()
        prediction_c = net(train_x)
        loss_c = loss_func(prediction_c, train_y)
        loss_c.backward()
        return loss_c
    optimizer.step(closure)
    if t % 1 == 0:
        #print ('epoch:{} {}'.format(t,loss.data[0]))
        grad_all = 0

        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all+=grad
        grad_norm = grad_all**0.5
        gra.append(grad_norm)
        if grad_zero > grad_norm:
            grad_zero = grad_norm
            weight = net.fc1.weight.data.numpy().flatten()
            cnt = 0
            for i in range(len(w)):
                if w[i] > 0:
                    cnt += 1
            ratio = cnt/len(w)
            print(ratio)
        total_w = sum(weight)
        if grad_norm < 0.1:
            losspoint.append(loss)
            cnt = 0
            for i in range(len(w)):draw.py
                if w[i] > 0:
                    cnt += 1
            ratio = cnt/len(w)
            min_ratio.append(ratio)
#import ipdb; ipdb.set_trace()

ep= list(range(0,1000))



test_x = torch.unsqueeze(torch.linspace(-10,10,100),dim=1)
test_y = torch.sin(test_x)
test_y = Variable(test_y)
test = net(Variable(test_x))
test_x= Variable(test_x)
#print(net.fc1.weight)
for i in range(len(losspoint)):
    losspoint[i] = losspoint[i].data
print (losspoint)
print (min_ratio)
plt.cla()
plt.title('sin min_ratio ')
plt.xlabel('min_ratio')
plt.ylabel('loss')
plt.scatter(min_ratio, losspoint)
#plt.scatter(test_x.data.numpy(), test_y.data.numpy())
#plt.plot(test_x.data.numpy(), test.data.numpy(), 'r-', lw=5)
#plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
plt.savefig('sin_min_ratio.png')

