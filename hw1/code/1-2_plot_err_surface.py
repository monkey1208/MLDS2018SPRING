from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv
import sys

data = []
loss = []
path_data = []
path_loss = []

for index, row in enumerate(csv.reader(open(sys.argv[1],'r'))):
	data.append([row[1],row[2]])
	loss.append(row[0])
	if index%101 == 1:
		path_data.append([row[1],row[2]])
		path_loss.append(row[0])
data = np.array(data,dtype=float)
loss = np.array(loss,dtype=float)
path_data = np.array(path_data,dtype=float)
path_loss = np.array(path_loss,dtype=float)
loss = np.log(loss)
path_loss = np.log(path_loss)

cm = plt.cm.get_cmap('RdYlBu')

fig = plt.figure()
#ax = Axes3D(fig)
plt.scatter(data[:,0], data[:,1], c=loss, cmap=cm, s=1)
plt.plot(path_data[:,0],path_data[:,1])
plt.colorbar()
plt.show()
