import os

import matplotlib
matplotlib.use('Agg') # NOQA
import matplotlib.pyplot as plt
import numpy as np


class Draw():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.counter = 0
    def plot_learning_curve(self, rewards, title, step=30):
        y = []
        for i in range(0, len(rewards), step):
            y.append(np.average(rewards[i:i+step]))
        x = np.arange(1, len(y)+1)
        plt.figure(self.counter)
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("# of time steps")
        plt.ylabel("average reward in last {} episodes".format(step))
        plt.savefig(os.path.join(self.output_dir, '{}.png'.format(title)))

        self.counter += 1
