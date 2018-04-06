import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from sklearn.decomposition import PCA

BATCH_SIZE = 15000
PRINT_FREQ = 1000
kwargs = {}
weight = []
weight_whole = []
loss_list = []
acc_list = []

def param_count(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('params:', params)

class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel,self).__init__()
		self.conv1 = nn.Sequential(
				nn.Conv2d(1,28,5),
				nn.ReLU(),
				nn.MaxPool2d(2)
		)
		self.prediction = nn.Linear(28*12*12,10,bias= True)
	def forward(self,x):
		x = self.conv1(x)
		x = x.view(x.size(0), -1)
		x = self.prediction(x)
		output = F.log_softmax(x, dim=1)
		return output

class CNNDeepModel(nn.Module):
	def __init__(self):
		super(CNNDeepModel,self).__init__()
		self.conv1 = nn.Sequential(
				nn.Conv2d(1,16,5),
				nn.ReLU(),
				nn.MaxPool2d(2)
		)
		self.conv2 = nn.Sequential(
				nn.Conv2d(16,16,5),
				nn.ReLU(),
				nn.MaxPool2d(2)
		)
		self.prediction = nn.Linear(16*4*4,10,bias=True)

	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x =  x.view(x.size(0), -1)
		x = self.prediction(x)
		output = F.log_softmax(x, dim=1)
		return output


test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
						])),batch_size=BATCH_SIZE, shuffle=True, **kwargs)
def train(train_loader,model,epoch):
	global weight, loss_list, test_loader, acc_list
	for i in range(int(sys.argv[1])):
		optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
		loss_mem = 0
		acc_mem = 0
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data), Variable(target)
			output = model(data)
			optimizer.zero_grad()
			loss = F.nll_loss(output, target)
			loss.backward()
			optimizer.step()
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					i, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.data[0]))
			loss_mem =loss.data[0]
		params = list(model.prediction.parameters())
		#if i%2==0:
		w = params[0].data.numpy()
		w = w.flatten()
		weight.append(w)
		params_all = list(model.parameters())
		w = params_all[0].data.numpy()
		w = w.flatten()
		weight_whole.append(w)
		acc_mem = test(test_loader,model)
		acc_list.append(acc_mem)
		loss_list.append(loss_mem)
		#output_file=open("model_"+str(epoch)+".pkl","wb")
		#for param in model.prediction.parameters():
		#	print(param.data)
		#torch.save(model.state_dict(),output_file)

def test(train_loader,model):
	correct = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		output = model(data)
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
	print("accuracy:"+str(correct/len(train_loader.dataset)))
	return correct/len(train_loader.dataset)

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
						transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
						])),batch_size=BATCH_SIZE, shuffle=True, **kwargs)

model = CNNModel()
model2 = CNNDeepModel()
isDeep = False
if sys.argv[3]=='deep':
	isDeep = True
'''for param in model.parameters():
	print(param)'''
for i in range(int(sys.argv[2])):
	print("Start training "+str(i)+"----------------------------------------------")
	if isDeep:
		model = CNNDeepModel()
	else:
		model = CNNModel()
	train(train_loader,model,i)
	print("End training "+str(i)+"------------------------------------------------")
weight = np.array(weight)
weight_whole = np.array(weight_whole)
print(weight.shape)
pca = PCA(n_components=2)
pca_reduction = pca.fit_transform(weight)
pca2 = PCA(n_components=2)
pca2_reduction = pca2.fit_transform(weight_whole)
print("pca_reduction: \n"+str(pca_reduction))
savepath = "loss_acc.csv"
if isDeep:
	savepath = "loss_acc_deep.csv"
writer = csv.writer(open(savepath,'w'),lineterminator='\n')
for row,row2,acc,loss in zip(pca_reduction,pca2_reduction,acc_list, loss_list):
	writer.writerow([row[0],row[1],row2[0],row2[1],acc,loss])

