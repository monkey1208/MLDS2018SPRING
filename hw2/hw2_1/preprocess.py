import os
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

def readfeat(train=True):
    path = checkconfig()
    data = []
    features = []
    ids = []
    idpath = 'training_id.txt' if train else 'testing_id.txt'
    with open(path+'/MLDS_hw2_1_data/'+idpath, 'r') as f:
        for line in f:
            video = line.strip('\n')
            ids.append(video)
            data.append(video+'.npy')
    featurepath = 'training_data' if train else 'testing_data'
    i = 0
    for d in data:
        i += 1
        feat = np.load(path+'/MLDS_hw2_1_data/'+featurepath+'/feat/'+d)
        features.append(feat)
        #print("\rprocessing features {0:.2f}%...  \r".format((i*100)/1450 , end="", flush=True))
        if train:
            sys.stdout.write("\rprocessing features {0:.2f}%...  ".format((i*100)/1450 ))
        else:
            sys.stdout.write("\rprocessing features {0:.2f}%...  ".format(i))
    print('')
    return np.array(features), ids

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

def label2onehot(label, limit=100):
    v = load_obj('vocabindex')
    onehot = []
    i = 1
    for video in label:
        onehot.append([])
        tmp = 0
        for sentence in video:
            tmp += 1
            if tmp > limit:
                break
            onehot[-1].append([])
            sentence = sentence.lower().replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace(',', '').replace('\"', '')
            words = sentence.split(' ')
            onehot[-1][-1].append(v['<start>'])
            for word in words:
                try:
                    onehot[-1][-1].append(v[word])
                except:
                    onehot[-1][-1].append(v['<unk>'])
            onehot[-1][-1].append(v['<end>'])
        print("\rprocessing label {0:.2f}%...  ".format((100 * i) / 1450), end="", flush=True)
        i += 1
    print('')
    return np.array(onehot), len(v)

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
            try:
                onehot[-1].append(v[word])
            except:
                onehot[-1].append(v['<unk>'])
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
    save_obj(v, 'vocabcount')
        #labels.append(label['caption'])

def label(threshold=0):
    v1 = {}
    v2 = {}
    v = load_obj('vocabcount')
    i = 1
    for word in v:
        if v[word] < threshold:
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
    vocab()
    label(3)
#readfeat()
#readlabel()

