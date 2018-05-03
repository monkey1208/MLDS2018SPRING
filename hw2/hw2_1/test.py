import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pack_padded_sequence
import random
import preprocess
from model import EncoderRNN, DecoderRNN, VanillaDecoderRNN, BAttnDecoderRNN
import ipdb
from tqdm import tqdm

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):
    fout = open(args.output, 'w')
    maxlen = 25
    num_layers = 4
    features, ids = preprocess.readfeat(False)
    v = preprocess.load_obj('vocabindex')
    v_size = len(v)
    print('feature shape =', features.shape)
    encoder = EncoderRNN(4096, 512, num_layers=num_layers, bidirectional=True)
    #decoder = VanillaDecoderRNN(512, v_size, num_layers=1)
    decoder = BAttnDecoderRNN(512, v_size, num_layers=num_layers)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))
    print(encoder)
    print(decoder)

    i = 0
    for feature in features:
        images = to_var(torch.Tensor(feature.reshape(1,feature.shape[0],feature.shape[1])))
        encoder_o, encoder_hidden = encoder(images)
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
            for key, indx in v.items():
                if indx == maxkey:
                    length += 1
                    word = key
                    if length > maxlen:
                        word = '.'
                        flag = False
                    if word == '<end>' or word == '<pad>':
                        flag = False
                    elif word == '<unk>':
                        continue
                    else:
                        if word == '.':
                            #sentence = sentence[:-1]
                            #sentence += word
                            flag = False
                        else:
                            if len(words) == 0:
                                word = word.title()
                            #sentence += word + ' '
                            if len(words) == 0 or word != words[-1]:
                                words.append(word)
                    break
        sentence = ' '.join(words)
        print(sentence)
        fout.write(ids[i]+','+sentence+'\n')
        i += 1
    fout.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True, help='encoder model')
    parser.add_argument("--decoder", type=str, required=True, help='decoder model')
    parser.add_argument("--output", type=str, required=True, help='output file')
    args = parser.parse_args()
    main(args)
