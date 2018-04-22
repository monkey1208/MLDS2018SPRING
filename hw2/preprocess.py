import os
import json
import numpy as np
import pickle
import random

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

def readfeat():
    path = checkconfig()
    data = []
    features = []
    with open(path+'/MLDS_hw2_1_data/training_id.txt', 'r') as f:
        for line in f:
            data.append(line.strip('\n')+'.npy')
    i = 0
    for d in data:
        i += 1
        feat = np.load(path+'/MLDS_hw2_1_data/training_data/feat/'+d)
        features.append(feat)
        print("\rprocessing features {0:.2f}%...  ".format((100*i)/1450) , end="", flush=True)
    print('')
    return np.array(features)

def readlabel():
    path = checkconfig()
    labels = []
    with open(path+'/MLDS_hw2_1_data/training_label.json', 'r') as f:
        text = f.read()
        l = json.loads(text.replace('\n',''))
    for label in l:
        labels.append(label['caption'])
        #print(label['caption'])
    return np.array(labels)

def label2onehot(label):
    v = load_obj('vocabindex')
    onehot = []
    i = 1
    for video in label:
        onehot.append([])
        for sentence in video:
            onehot[-1].append([])
            sentence = sentence.replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace(',', '').replace('\"', '')
            words = sentence.split(' ')
            onehot[-1][-1].append(v['<start>'])
            for word in words:
                onehot[-1][-1].append(v[word])
            onehot[-1][-1].append(v['<end>'])
        print("\rprocessing label {0:.2f}%...  ".format((100 * i) / 1450), end="", flush=True)
        i += 1
    print('')
    return np.array(onehot)

def label2onehot_single_sentence(label):
    v = load_obj('vocabindex')
    onehot = []
    i = 1
    for video in label:
        onehot.append([])
        sentence = video[random.randrange(1, len(video), 1)]

        sentence = sentence.lower().replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace(',', '').replace('\"', '')
        words = sentence.split(' ')
        onehot[-1].append(v['<start>'])
        for word in words:
            onehot[-1].append(v[word])
        onehot[-1].append(v['<end>'])
        print("\rprocessing label {0:.2f}%...  ".format((100 * i) / 1450), end="", flush=True)
        i += 1
    print('')
    return np.array(onehot), len(v)

def vocab():
    path = checkconfig()
    v = {}
    #indx = 0
    with open(path + '/MLDS_hw2_1_data/training_label.json', 'r') as f:
        text = f.read()
        l = json.loads(text.replace('\n', ''))
    for labels in l:
        for label in labels['caption']:
            label = label.replace('.', ' .').replace('!',' !').replace('?', ' ?').replace(',','').replace('\"', '')
            label = label.lower()
            words = label.split(' ')
            for word in words:
                if word not in v:
                    v[word] = 0
                v[word] += 1
            #print(words)
    print(v)
    v['<pad>'] = 1000
    v['<start>'] = 1000
    v['<end>'] = 1000
    v['<unk>'] = 1000
    save_obj(v, 'vocabcount')
        #labels.append(label['caption'])

def label(threshold=0):
    v1 = {}
    v = load_obj('vocabcount')
    i = 1
    for word in v:
        if v[word] < threshold:
            continue
        v1[word] = i
        i += 1
    save_obj(v1, 'vocabindex')
    print(v1)
    print(len(v1))

    #return np.array(labels)
if __name__ == "__main__":
    vocab()
    label()
#readfeat()
#readlabel()

