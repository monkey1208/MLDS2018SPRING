import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plot():
    def __init__(self):
        self.x = []
        self.y = []
        self.x2 = []
        self.y2 = []
    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def add2(self, x, y):
        self.x2.append(x)
        self.y2.append(y)

    def plot(self, x1, x2, y1, y2):
        dot1, = plt.plot(self.x, self.y, 'ro')
        dot2, =plt.plot(self.x2, self.y2, 'go')
        plt.xticks(np.arange(x1,x2+1,1.0))
        plt.yticks(np.arange(y1,y2,0.1))
        plt.legend([dot1, dot2], ['shallow', 'deep'])
        plt.show()

def test():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2.114, 1.8875, 1.8194, 1.7945, 1.7771, 1.7525, 1.736, 1.7241, 1.6981, 1.6811]
    y2 = [1.9401, 1.7085, 1.5428, 1.4127, 1.3075, 1.2464, 1.2005, 1.1576, 1.124, 1.0896]
    # z = [1,1,3]
    # plt.scatter(x, y, c=np.array(z),s=10)
    # plt.show()
    line_x, = plt.plot(x, y, 'ro', label='shallow')
    line_x2, = plt.plot(x, y2, 'go', label='deep')
    plt.xticks(np.arange(1, 11, 1.0))
    plt.yticks(np.arange(0.8, 2.2, 0.1))
    plt.legend([line_x, line_x2], ['shallow', 'deep'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('CIFAR10')
    plt.show()

def test2():
    fig, ax = plt.subplots()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2.114, 1.8875, 1.8194, 1.7945, 1.7771, 1.7525, 1.736, 1.7241, 1.6981, 1.6811]
    y2 = [1.9401, 1.7085, 1.5428, 1.4127, 1.3075, 1.2464, 1.2005, 1.1576, 1.124, 1.0896]
    #line, = ax.plot(x, y, lw=2)
    ax.scatter(x,y)
    for i, txt in enumerate(y2):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

def hw1_3_1():
    data = pd.read_csv('/Users/Yang/Documents/junior/MLDS/hw1/1-3-1.csv')
    #print(data)
    epoch = data['epoch'].get_values()
    train_l = data['train_loss'].get_values()
    test_l = data['test_loss'].get_values()
    fig, ax = plt.subplots()
    line1, = ax.plot(epoch, train_l)
    line2, = ax.plot(epoch, test_l)
    plt.legend([line1, line2], ['train', 'test'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def hw1_3_2():
    data = pd.read_csv('~/Documents/junior/MLDS/hw1/hw1-3-2.csv')
    data = data.sort_values(by=['params'])
    print(data)
    params = np.array(data['params'].get_values())
    train_l = np.array(data['train_loss'].get_values())
    train_a = np.array(data['train_accuracy'].get_values())
    test_l = np.array(data['test_loss'].get_values())
    test_a = np.array(data['test_accuracy'].get_values())
    fig, ax = plt.subplots()
    line1 = ax.scatter(params, train_l)
    line2 = ax.scatter(params, test_l)
    #line1, = ax.plot(params, train_l)
    #line2, = ax.plot(params, test_l)
    plt.legend([line1, line2], ['train', 'test'])
    plt.xlabel('params')
    plt.ylabel('loss')
    plt.title('loss')
    plt.show()
    fig, ax = plt.subplots()
    line3 = ax.scatter(params, train_a)
    line4 = ax.scatter(params, test_a)
    plt.legend([line3, line4], ['train', 'test'])
    plt.xlabel('params')
    plt.ylabel('accuracy')
    plt.title('accuracy')
    plt.show()

def hw1_3_3_1():
    data = pd.read_csv('1-3-3-1.csv')
    alpha = data['alpha'].values
    train_accuracy = data['train_accuracy'].values
    train_loss = data['train_loss'].values
    test_accuracy = data['test_accuracy'].values
    test_loss = data['test_loss'].values
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    line1, = ax.plot(alpha, train_loss, color='red', linestyle=':')
    line2, = ax.plot(alpha, test_loss, linestyle='-', color='red')
    ax.set_xlabel('alpha')
    ax.set_ylabel('entropy_loss')
    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    ax2.plot(alpha, train_accuracy, color='blue', linestyle=':')
    ax2.plot(alpha, test_accuracy, linestyle='-', color='blue')
    ax2.set_ylabel('accuracy')
    ax2.yaxis.tick_right()
    ax2.yaxis.label.set_color('blue')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color='blue', labelcolor='blue')
    plt.legend([line2, line1], ['train', 'test'])
    plt.title('batch size')
    plt.show()

def hw1_3_3_2():
    data = pd.read_csv('hw1-3-3-2.csv')
    bsize = data['batch_size'].values
    train_accuracy = data['train_accuracy'].values
    train_loss = data['train_loss'].values
    test_accuracy = data['test_accuracy'].values
    test_loss = data['test_loss'].values
    norm = data['norm'].values
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    line1, = ax.plot(bsize, train_accuracy, color='red', linestyle='-')
    line2, = ax.plot(bsize, test_accuracy, color='red', linestyle=':')

    ax.set_xlabel('batch_size')
    ax.set_ylabel('accuracy')
    ax.tick_params(axis='y', colors='red')
    ax.xaxis.set_visible(False)
    ax.yaxis.label.set_color('red')
    #ax2.plot(, train_accuracy, color='blue', linestyle=':')
    #ax2.plot(alpha, test_accuracy, linestyle='-', color='blue')
    line3, = ax2.plot(bsize, norm, linestyle='-', color='blue')
    ax2.set_ylabel('sensitive')
    ax2.yaxis.tick_right()
    ax2.yaxis.label.set_color('blue')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color='blue', labelcolor='blue')
    plt.legend([line1, line2, line3], ['train', 'test', 'sensitive'])
    plt.title('accuracy')
    plt.xscale('log')
    plt.show()

def main():
    hw1_3_3_2()

if __name__ == '__main__':
    main()