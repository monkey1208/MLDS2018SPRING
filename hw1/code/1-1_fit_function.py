import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

def sign(x): return 1 if x >= 0 else -1

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(in_features=1, out_features=200, bias=True)
		self.fc2 = nn.Linear(in_features=200, out_features=1, bias=True)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
class DeepNet(nn.Module):
	def __init__(self):
		super(DeepNet,self).__init__()
		self.fc1 = nn.Linear(in_features=1,out_features=10,bias=True)
		self.fc2 = nn.Linear(in_features=10,out_features=27,bias=True)
		self.fc3 = nn.Linear(in_features=27,out_features=10,bias=True)
		self.fc4 = nn.Linear(in_features=10,out_features=1,bias=True)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x
class SuperDeepNet(nn.Module):
	def __init__(self):
		super(SuperDeepNet,self).__init__()
		self.fc1 = nn.Linear(in_features=1,out_features=8,bias=True)
		self.fc2 = nn.Linear(in_features=8,out_features=16,bias=True)
		self.fc3 = nn.Linear(in_features=16,out_features=16,bias=True)
		self.fc4 = nn.Linear(in_features=16,out_features=8,bias=True)
		self.fc5 = nn.Linear(in_features=8,out_features=4,bias=True)
		self.fc6 = nn.Linear(in_features=4,out_features=1,bias=True)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = self.fc6(x)
		return x
def train(is_deep,is_super, model_path, epoch, data_path, loss_path):
	loss_value = []
	x, y = readfile(data_path)
	x = torch.from_numpy(x).float().unsqueeze(1)
	y = torch.from_numpy(y).float().unsqueeze(1)
	x = Variable(x)
	y = Variable(y)

	net = Net()
	if is_deep:
		net =DeepNet()
	if is_super:
		net = SuperDeepNet()
	print(net)
	param_count(net)

	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	min = -1
	for i in range(epoch):
		output = net(x)
		crit = nn.MSELoss()
		loss = crit(output,y)
		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()
		#print("loss:"+str(loss.data[0]))
		if min==-1 or loss.data[0]<min:
			output_file=open(model_path,"wb")
			torch.save(net.state_dict(),output_file)
			min = loss.data[0]
		loss_value.append(loss.data[0])
	writer = csv.writer(open(loss_path,'w'),lineterminator='\n')
	for i,row in enumerate(loss_value):
		if i%100 ==0:
			writer.writerow([i/100,row])

def test(is_deep, model_path,epoch):

	test_value = [i for i in range(-10,10)]
	net = Net()
	if is_deep == 0:
		net = DeepNet()
	elif is_deep == 2:
		net = SuperDeepNet()


	test_value = np.array(test_value,dtype=float)
	test_value = torch.from_numpy(test_value).float().unsqueeze(1)
	test_value = Variable(test_value)
	input_file = open(model_path,"rb")
	net.load_state_dict(torch.load(input_file))
	prediction = net(test_value)
	#for i, j in zip():
	#	out.append([i.data[0],j.data[0]])
	data = [i.data[0] for i in test_value]
	pred = [i.data[0] for i in prediction]
	return data, pred

def loss_plot(x1, y1, x2, y2, x3, y3):

	y1 = np.log10(y1)
	y2 = np.log10(y2)
	y3 = np.log10(y3)
	deep_plot, = plt.plot(x1,y1,label = 'model1(deep)', color= 'red')
	shallow_plot, = plt.plot(x2,y2, label = 'model2(shallow)', color= 'blue')
	deeper_plot, = plt.plot(x3,y3,label='model3(deeper)', color= 'lime')
	plt.legend()
	plt.show()

def plot(x1, y1, x2, y2, x3, y3):

	deep_plot, = plt.plot(x1,y1,label='model1(deep)', color= 'red')
	shallow_plot, = plt.plot(x2,y2,label='model2(shallow)', color= 'blue')
	deeper_plot, = plt.plot(x3,y3,label='model3(deeper)', color= 'lime')
	x_std = [i*0.01 for i in range(-1000,1000)]
	y_std = [sign(math.sin(i*0.01)) for i in range(-1000,1000)]
	std_plot, = plt.plot(x_std,y_std,label='function', color= 'grey')
	plt.legend()
	plt.show()

def readfile(data_path):
	f = open(data_path,'r')
	file_data = []
	for row in csv.reader(f):
		file_data.append(row)
	file_data = np.array(file_data,dtype=float)
	data = file_data[:,2]
	label = file_data[:,3]
	return data,label

def param_count(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('params:', params)

#def BuildDeepModel():


def main(opts):
	if opts.train:
		print("Start training -------------------------------------------------")
		train(opts.deep, opts.superdeep, opts.model_path, opts.epoch,opts.data_path,opts.loss_path)
	if opts.test:
		x_deep, y_deep = test(0, opts.model_path, opts.epoch)
		x_shal, y_shal = test(1, opts.model2_path, opts.epoch)
		x_sup , y_sup  = test(2, opts.model3_path, opts.epoch)
		index, loss_deep = readfile("loss_deep.csv")
		index1, loss_shallow = readfile("loss_shallow.csv")
		index2, loss_deeper = readfile("loss_deeper.csv")

		plot(x_deep, y_deep, x_shal, y_shal, x_sup, y_sup)
		loss_plot(index, loss_deep, index1, loss_shallow, index2, loss_deeper)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-train',action='store_true', default=False,dest='train',help='Train')
	parser.add_argument('-test',action='store_true', default=False,dest='test',help='Test')
	parser.add_argument('-deep',action='store_true', default=False,dest='deep',help='Use deep model')
	parser.add_argument('-superdeep',action='store_true', default=False,dest='superdeep',help='Use deep model')
	parser.add_argument('--model_path', type=str,default='md_deep.pkl',dest='model_path',help='model name')
	parser.add_argument('--model2_path', type=str,default='md_shallow.pkl',dest='model2_path',help='model name')
	parser.add_argument('--model3_path', type=str,default='md_deeper.pkl',dest='model3_path',help='model name')
	parser.add_argument('--data_path', type=str,default='data.csv',dest='data_path',help='data name')
	parser.add_argument('--loss_path', type=str,default='loss.csv',dest='loss_path',help='loss name')

	parser.add_argument('--epoch',type=int,default='1000',dest='epoch',help='epoch')
	opts = parser.parse_args()
	main(opts)

