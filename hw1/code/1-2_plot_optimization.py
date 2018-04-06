import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

data = [[] for i in range(int(sys.argv[2]))]
colors = ['red','orange','pink','green','blue','black','purple','grey']

num = -1
for index,row in enumerate(csv.reader(open(sys.argv[1],"r"))):
	if index%15==0:
		num =num +1
	data[num].append(row)

fig, ax = plt.subplots()
for i in range(int(sys.argv[2])):
	temp = data[i]
	temp = np.array(temp,dtype= float)
	print(temp.shape)
	loss = np.around(temp[:,4],decimals=2)
	ax.scatter(temp[:,0],temp[:,1], s=20, label="model"+str(i),
			alpha=0, edgecolors='none')
	for j, txt in enumerate(loss):
		ax.annotate(txt, (temp[j][0],temp[j][1]), color = colors[i])

ax.legend(bbox_to_anchor=(1,1), loc='center left')
ax.grid(True)
plt.show()
fig, ax = plt.subplots()
for i in range(int(sys.argv[2])):
	temp = data[i]
	temp = np.array(temp,dtype= float)
	loss = np.around(temp[:,4],decimals=2)
	ax.scatter(temp[:,2],temp[:,3], s=20, label="model"+str(i),
			alpha=0, edgecolors='none')
	for j, txt in enumerate(loss):
		ax.annotate(txt, (temp[j][0],temp[j][1]), color = colors[i])

ax.legend(bbox_to_anchor=(1,1), loc='center left')
ax.grid(True)
plt.show()
