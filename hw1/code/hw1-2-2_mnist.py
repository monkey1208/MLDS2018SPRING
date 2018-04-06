import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

print
'==>>> total trainning batch number: {}'.format(len(train_loader))
print
'==>>> total testing batch number: {}'.format(len(test_loader))



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, 1)
        self.conv2 = nn.Conv2d(3, 5, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 5, 5)
        self.fc2 = nn.Linear(5, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    #def backward(self, hook):


    def name(self):
        return "LeNet"


## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

ceriation = nn.CrossEntropyLoss()

#w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
#w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
gra = []
cnt = 0
ep = list(range(0,6000))
for epoch in range(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        grad_all = 0.0
        cnt+=1
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all+=grad
        grad_norm = grad_all**0.5
        print(grad_norm)
        gra.append(ave_loss)
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, targe = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(epoch, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))

torch.save(model.state_dict(), model.name())
plt.cla()
plt.plot(ep, gra)
plt.title('Mnist loss')
plt.xlabel('iteration')
plt.ylabel('loss')
#plt.scatter(test_x.data.numpy(), test_y.data.numpy())
#plt.plot(test_x.data.numpy(), test.data.numpy(), 'r-', lw=5)
#plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
plt.savefig('mnist_loss.png')
