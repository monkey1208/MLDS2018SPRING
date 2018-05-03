import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pack_padded_sequence
import random
import sys
import preprocess
from model import EncoderRNN, DecoderRNN, VanillaDecoderRNN, BAttnDecoderRNN
import ipdb
from tqdm import tqdm
import dataloader

def getDataLoader(data, label, batch_size=64, num_workers=4, single_sentence=False):
    if single_sentence:
        dataset = dataloader.Dataset(images=data,labels=label)
    else:
        dataset = dataloader.DatasetMultilabel(images=data,labels=label)
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

def main(args):
    teacher_forcing_ratio = 0.5
    schedule_sampling_ratio = 0.7
    num_layers = 2
    features, _ = preprocess.readfeat()
    #labels, v_size = preprocess.label2onehot_single_sentence(preprocess.readlabel())
    labels, v_size = preprocess.label2onehot(preprocess.readlabel(), limit=12)
    i2v = preprocess.load_obj('index2vocab')
    print('feature shape =', features.shape)
    print('label shape =', labels.shape)
    dataloader = getDataLoader(features, labels, single_sentence=False, batch_size=128)
    encoder = EncoderRNN(4096, 512, num_layers=num_layers, bidirectional=True)
    #decoder = VanillaDecoderRNN(512, v_size, num_layers=1)
    decoder = BAttnDecoderRNN(512, v_size, num_layers=num_layers)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    print(encoder)
    print(decoder)
    c = nn.CrossEntropyLoss()
    #c = nn.NLLLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0003)
    epochs = 16
    for epoch in range(epochs):
        losses = []
        desc = 'Epoch [{}/{}]'.format(epoch + 1, epochs)
        train_loss = []
        for images, inputs, targets, lengths in tqdm(dataloader, desc=desc):
            images, inputs, targets = to_var(images), to_var(inputs), to_var(targets)
            batch_size, caption_len = inputs.size()[0], inputs.size()[1]
            loss = 0
            encoder.zero_grad()
            decoder.zero_grad()
            encoder_output, encoder_hidden = encoder(images)
            decoder_hidden = encoder_hidden[:num_layers]
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
        if (epoch+1) % 2 == 0:
             torch.save(encoder.state_dict(), args.model+'/encoder_epoch{}.pt'.format(epoch+1))
             torch.save(decoder.state_dict(), args.model+'/decoder_epoch{}.pt'.format(epoch+1))
        print('loss={:.4f}'.format(np.average(losses)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--learntype", type=str, default='teacher_forcing')
    args = parser.parse_args()
    main(args)
