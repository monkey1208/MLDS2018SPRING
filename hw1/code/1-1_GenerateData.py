import sys
import csv
import random
import matplotlib.pyplot as plt
import math

UPPER_BOUND = 10
LOWER_BOUND = -10

def sign(x): return 1 if x >= 0 else -1

writer = csv.writer(open(sys.argv[1],'w'),lineterminator='\n')

for i in range(int(sys.argv[2])):
	x = random.uniform(LOWER_BOUND,UPPER_BOUND)
	writer.writerow([x,sign(math.sin(x))])
x_std = [i for i in range(-LOWER_BOUND,UPPER_BOUND)]
y_std = [sign(math.sin(i)) for i in range(-LOWER_BOUND,UPPER_BOUND)]
plt.plot(x_std,y_std)
plt.show()
