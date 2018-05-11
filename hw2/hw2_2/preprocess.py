import os
import operator
import json
import numpy as np
import pickle
import random
import sys

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def checkconfig():
    with open('config.json', 'r') as f:
        cfg = json.loads(f.read().replace('\n',''))
    #os.chdir(cfg['path'])
    return cfg['path']

def label2onehot(path):
    v = load_obj('vocabindex')
    onehot = []
    i = 1
    with open(path, 'r') as f:
        for line in f:
            onehot.append([])
            words = line.strip('\n').split(' ')
            onehot[-1].append(v['<start>'])
            for word in words:
                if word == '+++$+++':
                    onehot[-1][0] = -1
                    break
                try:
                    onehot[-1].append(v[word])
                except:
                    onehot[-1].append(v['<unk>'])
            onehot[-1].append(v['<end>'])
    return np.array(onehot), len(v)

def vocab(path):
    v = {}
    #indx = 0
    with open(path + '/clr_conversation.txt', 'r') as f:
        i = 0
        for line in f:
            words = line.strip('\n').split(' ')
            for word in words:
                if word == '+++$+++':
                    i += 1
                    continue
                if word not in v:
                    v[word] = 0
                v[word] += 1
    print(v)
    print(len(v))
    print(i)
    save_obj(v, 'vocabcount')

def label(threshold=0):
    v1 = {}
    v2 = {}
    v = load_obj('vocabcount')
    i = 1
    for word in v:
        if v[word] <= threshold:
            continue
        v1[word] = i
        v2[i] = word
        i += 1
    v1['<pad>'] = 0
    v2[0] = '<pad>'
    v1['<start>'] = i
    v2[i] = '<start>'
    v1['<end>'] = i+1
    v2[i+1] = '<end>'
    v1['<unk>'] = i+2
    v2[i+2] = '<unk>'
    save_obj(v1, 'vocabindex')
    save_obj(v2, 'index2vocab')
    print(v1)
    print(len(v1))

    #return np.array(labels)
if __name__ == "__main__":
    #vocab('/4t/ylc/MLDS/hw2_2/')
    #label(15)
    label(20)
    #label(87)

