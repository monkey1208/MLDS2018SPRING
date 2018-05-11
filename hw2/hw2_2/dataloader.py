import torch
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        indx = 0
        self.train_sentence = []
        for s in range(len(sentences)):
            if sentences[s][0] == -1 or (s != len(sentences) - 1 and sentences[s+1][0] == -1) or s == len(sentences) - 1:
                indx += 1
                continue
            else:
                self.train_sentence.append(indx)
            indx += 1
        self.train_sentence = np.array(self.train_sentence)

    def __getitem__(self, index):
        label = np.array(self.sentences[self.train_sentence[index]+1])
        sentence = np.array(self.sentences[self.train_sentence[index]])
        return torch.LongTensor(sentence), torch.LongTensor(label)

    def __len__(self):
        return len(self.train_sentence)
class TestDataset(data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, index):
        return torch.LongTensor(self.sentences[index]), index

    def __len__(self):
        return len(self.sentences)

def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sentences, label = zip(*data)
    lengths = [len(sent) for sent in sentences]
    #s = torch.zeros(len(sentences), max(lengths)+1).long()
    s = torch.zeros(len(sentences), max(lengths)+1).long()
    for i, sent in enumerate(sentences):
        end = lengths[i]
        s[i, :end] = sent[:end]
    labellengths = [len(lab)-1 for lab in label]
    targets = torch.zeros(len(label), max(labellengths)).long()
    inputs = torch.zeros(len(label), max(labellengths)).long()
    for i, lab in enumerate(label):
        end = labellengths[i]
        targets[i, :end] = lab[1:end+1]
        inputs[i, :end] = lab[:end]
    return s, inputs, targets, lengths

def test_collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sentences, index = zip(*data)
    lengths = [len(sent) for sent in sentences]
    s = torch.zeros(len(sentences), max(lengths)+1).long()
    for i, sent in enumerate(sentences):
        end = lengths[i]
        s[i, :end] = sent[:end]
    return s, lengths, index
