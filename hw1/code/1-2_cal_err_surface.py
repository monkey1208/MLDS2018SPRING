import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

weight = []
loss_list = []
grad = 0

def param_count(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('params:', params)

class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.fc1 = nn.Linear(1,16,bias= True)
		self.prediction = nn.Linear(16,1,bias= True)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = self.prediction(x)
		return x
def train(model, data, target, sec):
	global grad
	if not sec:
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		output = model(data)
		optimizer.zero_grad()
		loss = F.mse_loss(output, target)
		loss.backward()
		optimizer.step()
	else:
		optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
		def closure():
			global grad
			output = model(data)
			loss = F.mse_loss(output, target)
			grad = list(torch.autograd.grad(loss,model.parameters(),create_graph=True))
			optimizer.zero_grad()
			loss.backward()
			return loss
		optimizer.step(closure)
		output = model(data)
		loss = F.mse_loss(output,target)
	return loss.data[0]

def sample_model(model,data,target,num):
	global weight,loss_list, grad
	param = list(model.fc1.parameters())
	param2 = list(model.prediction.parameters())
	grad1 = grad[0].data.numpy()
	grad2 = grad[1].data.numpy()
	model2 = Model()
	w = param[0].data.numpy()
	w_t = np.copy(w)
	b = param[1].data.numpy()
	w2 = param2[0].data.numpy()
	w2_t = np.copy(w2)
	b2 = param2[1].data.numpy()
	w_ff = np.append(w,w2)
	w_ff = w_ff.flatten()
	weight.append(w_ff)
	for i in range(500):
		for index, row in enumerate(w):
			rand = random.uniform(0.80,1.20)*0.001
			w_t[index] = row - grad1[index]*rand
		for index, row in enumerate(w2):
			rand = random.uniform(0.8,1.2)
			w2_t[index] = row - grad2[index]*rand*0.001
		model2.fc1.weight = nn.Parameter(torch.from_numpy(w_t))
		model2.fc1.bias = nn.Parameter(torch.from_numpy(b))
		model2.prediction.weight = nn.Parameter(torch.from_numpy(w2_t))
		model2.prediction.bias = nn.Parameter(torch.from_numpy(b2))
		output = model2(data)
		loss = F.mse_loss(output,target)
		loss_list.append(loss.data[0])
		w_f = np.append(w_t,w2_t)
		w_f = w_f.flatten()
		weight.append(w_f)
		print("modified Loss{} :{:.6f}".format(i,loss.data[0]))

def main():
	global weight,loss_list
	data = np.random.uniform(-10,10,100)
	label = [i*i*i for i in data]
	label = np.array(label)

	data, label = data.reshape(100,1), label.reshape(100,1)
	data, label = Variable(torch.from_numpy(data).float(), requires_grad=True),Variable(torch.from_numpy(label).float())
	model = Model()
	for i in range(int(sys.argv[1])):
		loss = train(model,data,label,False)
		print("EPOCH:{}\tLOSS:{:.6f}".format(i,loss))
	for i in range(10):
		loss = train(model,data,label,True)
		loss_list.append(loss)
		print("SECOND:{}\tLOSS:{:.6f}".format(i,loss))
		sample_model(model,data,label,10)
	writer = csv.writer(open("tsne_error_surface.csv",'w'),lineterminator='\n')
	weight = np.array(weight)
	W_embedded = TSNE(n_components=2).fit_transform(weight)
	print(W_embedded.shape)
	for loss, w in zip(loss_list,W_embedded):
		writer.writerow([loss,w[0],w[1]])

if __name__ == "__main__":
	main()