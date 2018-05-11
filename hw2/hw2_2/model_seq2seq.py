import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pack_padded_sequence
import random
import json
import sys
import preprocess
from model import EncoderRNN, DecoderRNN, VanillaDecoderRNN, BAttnDecoderRNN, LAttnDecoderRNN
from tqdm import tqdm
import dataloader

def readconfig(path):
    with open(path, 'r') as f:
        cfg = json.loads(f.read().replace('\n',''))
    return cfg

def getDataLoader(data, batch_size=100, num_workers=4):
    dataset = dataloader.Dataset(sentences=data)
    loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=num_workers, 
        collate_fn=dataloader.collate_fn
    )
    return loader

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def index2sentence(s):
    v = preprocess.load_obj('index2vocab')
    for w in s:
        print(v[w.item()], end=' ')

def main(args):
    cfg = readconfig(args.config)
    teacher_forcing_ratio = 0.5
    schedule_sampling_ratio = cfg['schedule_sampling_ratio']
    dim = cfg['dim']
    num_layers = cfg['num_layers']
    bidirectional = cfg['bidirectional']
    lr = cfg['learning_rate']
    labels, v_size = preprocess.label2onehot('/4t/ylc/MLDS/hw2_2/clr_conversation.txt')
    labels = np.concatenate((labels, labels_2))
    print('label shape =', labels.shape)
    dataloader = getDataLoader(labels, batch_size=50)
    encoder = EncoderRNN(v_size, dim, num_layers=num_layers, bidirectional=bidirectional)
    #decoder = VanillaDecoderRNN(dim, v_size, num_layers=num_layers)
    #decoder = BAttnDecoderRNN(dim, v_size, num_layers=num_layers)
    decoder = LAttnDecoderRNN(dim, v_size, num_layers=num_layers)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    if args.encoder != None:
        encoder.load_state_dict(torch.load(args.encoder))
    if args.decoder != None:
        decoder.load_state_dict(torch.load(args.decoder))
    print(encoder)
    print(decoder)
    c = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)
    epochs = args.epochs
    for epoch in range(args.start_epoch, epochs):
        it = 0
        losses = []
        desc = 'Epoch [{}/{}]'.format(epoch + 1, epochs)
        train_loss = []
        for sentences, inputs, targets, lengths in tqdm(dataloader, desc=desc):
            it += 1
            sentences, inputs, targets = to_var(sentences), to_var(inputs), to_var(targets)
            batch_size, caption_len = inputs.size()[0], inputs.size()[1]
            loss = 0
            encoder.zero_grad()
            decoder.zero_grad()
            encoder_hidden = encoder.initHidden(num_layers, batch_size)
            encoder_output, encoder_hidden = encoder(sentences, lengths, encoder_hidden)
            decoder_hidden = encoder_hidden[:num_layers]
            encoder_output = pad_packed_sequence(encoder_output, batch_first=True)[0]
            decoder_outputs = torch.autograd.Variable(torch.zeros(batch_size, caption_len, v_size))
            if torch.cuda.is_available():
                decoder_outputs = decoder_outputs.cuda()
            
            #use teacher forcing or not
            if args.learntype == 'teacher_forcing':
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    for wordindex in range(caption_len):
                        inputword = inputs[:, wordindex]
                        output, decoder_hidden = decoder(inputword, decoder_hidden, encoder_output)
                        decoder_outputs[:, wordindex, :] = output
                else:
                    inputword = inputs[:, 0]
                    for wordindex in range(caption_len):
                        output, decoder_hidden = decoder(inputword, decoder_hidden, encoder_output)
                        maxkey = np.argmax(output.data, axis=1)
                        inputword = to_var(maxkey)
                        decoder_outputs[:, wordindex, :] = output
            #schedule sampling
            else:
                inputword = inputs[:, 0]
                for wordindex in range(caption_len):
                    output, decoder_hidden = decoder(inputword, decoder_hidden, encoder_output)
                    decoder_outputs[:, wordindex, :] = output
                    if random.random() < schedule_sampling_ratio and wordindex < caption_len - 1:
                        inputword = inputs[:, wordindex+1]
                    else:
                        maxkey = np.argmax(output.data, axis=1)
                        inputword = to_var(maxkey)
                        
            for i in range(batch_size):
                loss += c(decoder_outputs[i], targets[i])
            loss.backward()
            optimizer.step()
            losses.append(loss.data/batch_size)
            if (it) % 1000 == 0:
                if (it) % 5000 == 0:
                    torch.save(encoder.state_dict(), args.model+'/it/encoder_epoch{}_iteration{}.pt'.format(epoch+1, it))
                    torch.save(decoder.state_dict(), args.model+'/it/decoder_epoch{}_iteration{}.pt'.format(epoch+1, it))
                print('loss={:.4f}'.format(np.average(losses)))
        if (epoch+1) % 1 == 0:
             torch.save(encoder.state_dict(), args.model+'/encoder_epoch{}.pt'.format(epoch+1))
             torch.save(decoder.state_dict(), args.model+'/decoder_epoch{}.pt'.format(epoch+1))
        print('loss={:.4f}'.format(np.average(losses)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--learntype", type=str, default='teacher_forcing')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--config", type=str, default='config.json')
    
    args = parser.parse_args()
    main(args)
