import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pack_padded_sequence
import preprocess
from model import EncoderRNN, DecoderRNN, VanillaDecoderRNN, BAttnDecoderRNN, LAttnDecoderRNN
import ipdb
import dataloader
from gensim.models import word2vec
from gensim import models

def getDataLoader(data, batch_size=100, num_workers=4):
    dataset = dataloader.TestDataset(data)
    loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  
        num_workers=num_workers, 
        collate_fn=dataloader.test_collate_fn
    )
    return loader

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def getwv():
    i2v = preprocess.load_obj('index2vocab')
    model = models.Word2Vec.load('w2v200.model.bin')
    embedding = np.zeros((len(i2v), model.wv.vector_size))
    j = 0
    for i in range(len(i2v)):
        try:
            embedding[i] = model.wv[i2v[i]]
        except:
            print(i2v[i], 'not in w2v')
            j += 1
            continue
    print(j)
    return embedding

def main(args):
    fout = open(args.output, 'w')
    maxlen = 20
    num_layers = 1
    dim = 512
    quotes_set = set(['[',']','{','}','!','?','。'])
    sentences, v_size = preprocess.label2onehot(args.input)
    data = getDataLoader(sentences)
    v = preprocess.load_obj('vocabindex')
    i2v = preprocess.load_obj('index2vocab')
    print(sentences)
    v_size = len(v)
    print('sentences shape =', sentences.shape)
    we = getwv()
    encoder = EncoderRNN(v_size, dim, we, num_layers=num_layers, bidirectional=False)
    #decoder = VanillaDecoderRNN(dim, v_size, num_layers=num_layers)
    #decoder = BAttnDecoderRNN(dim, v_size, num_layers=num_layers)
    decoder = LAttnDecoderRNN(dim, v_size, we, num_layers=num_layers)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    print(encoder)
    print(decoder)
    i = 0
    for sentence in sentences:
        print(sentence)
        s = to_var(torch.LongTensor(sentence).view(1, -1))
        encoder_hidden = encoder.initHidden(num_layers, 1)
        encoder_o, encoder_hidden = encoder(s, [s.size(1)], encoder_hidden)
        encoder_o = pad_packed_sequence(encoder_o, batch_first=True)[0]
        decoder_hidden = encoder_hidden[:num_layers]
        inputword = v['<start>']
        flag = True
        sentence = ''
        words = []
        length = 0
        while flag:
            inputword = to_var(torch.LongTensor([inputword]).view(1,-1))
            output, decoder_hidden = decoder(inputword,decoder_hidden, encoder_o)
            maxkey = np.argmax(output[0].data)
            inputword = maxkey
            word = i2v[maxkey.item()]
            length += 1
            if length > maxlen:
                flag = False
            if word == '<end>' or word == '<pad>':
                flag = False
            elif word == '<unk>':
                continue
            else:
                if word == '.' or word == '。':
                    #sentence = sentence[:-1]
                    #sentence += word
                    flag = False
                else:
                    if word in words and word in quotes_set:
                        continue
                    if (len(words) == 0 or word != words[-1]):
                        words.append(word)
        sentence = ' '.join(words)
        print(sentence)
        fout.write(sentence+'\n')
        i += 1
    
    fout.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help='input file')
    parser.add_argument("--encoder", type=str, required=True, help='encoder model')
    parser.add_argument("--decoder", type=str, required=True, help='decoder model')
    parser.add_argument("--output", type=str, required=True, help='output file')
    args = parser.parse_args()
    main(args)
