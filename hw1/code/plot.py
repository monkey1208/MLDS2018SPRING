import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

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

def hw1_3_1():
    data = pd.read_csv('hw1-3-1.csv')
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
    data = pd.read_csv('hw1-3-2.csv')
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
    data = pd.read_csv('hw1-3-3-1.csv')
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
    cmd = sys.argv[1]
    if cmd == '1_3_1':
        hw1_3_1()
    elif cmd == '1_3_2':
        hw1_3_2()
    elif cmd == '1_3_3_1':
        hw1_3_3_1()
    elif cmd == '1_3_3_2':
        hw1_3_3_2()

if __name__ == '__main__':
    main()
