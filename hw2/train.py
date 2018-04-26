import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pack_padded_sequence
import random
import preprocess
from model import EncoderRNN, DecoderRNN
import ipdb
from tqdm import tqdm
import dataloader

def getDataLoader(data, label, batchsize=128, num_workers=2):
    dataset = dataloader.Dataset(images=data,labels=label)
    loader = Data.DataLoader(
                             dataset=dataset,  # torch TensorDataset format
                             batch_size=batchsize,  # mini batch size
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
    features = preprocess.readfeat()
    labels, v_size = preprocess.label2onehot_single_sentence(preprocess.readlabel())
    print('feature shape =', features.shape)
    print('label shape =', labels)
    dataloader = getDataLoader(features, labels)
    encoder = EncoderRNN(4096, 256)
    decoder = DecoderRNN(256, v_size)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    print(encoder)
    print(decoder)

criterion = nn.CrossEntropyLoss()
    c = nn.NLLLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params)
    epochs = 100
    loss = 0
    for epoch in range(epochs):
        desc = 'Epoch [{}/{}]'.format(epoch + 1, epochs)
        train_loss = []
        #encoder_hidden = encoder.initHidden()
        for images, inputs, targets, lengths in tqdm(dataloader, desc=desc):
            images, inputs, targets = to_var(images), to_var(inputs), to_var(targets)
            batch_size, caption_len = inputs.size()[0], inputs.size()[1]
            loss = 0
            encoder.zero_grad()
            decoder.zero_grad()
            encoder_o, encoder_hidden = encoder(images)
            decoder_hidden = encoder_hidden
            
            decoder_outputs = torch.autograd.Variable(torch.zeros(batch_size, caption_len, v_size))
            if torch.cuda.is_available():
                decoder_outputs = decoder_outputs.cuda()
        
            #use teacher forcing or not
            use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False
            #ipdb.set_trace()
            
            if use_teacher_forcing:
                outputs, decoder_hidden, batch = decoder(inputs, lengths, decoder_hidden)
                #ipdb.set_trace()
                test, _ = pad_packed_sequence((outputs, batch), batch_first=True)
                test = test.contiguous().view(-1, v_size)
                targets = targets.contiguous().view(-1)
                loss = c(test, targets)
                loss.backward()
                print(loss)
                optimizer.step()

    else:
        
        for di in range(caption_len):
            outputs, decoder_hidden = decoder(inputs, lengths, decoder_hidden)
                ipdb.set_trace()


torch.save(
           decoder.state_dict(),
           "decoder_model.pkl"
           )
    torch.save(
               encoder.state_dict(),
               "encoder_model.pkl"
               )

#ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("-")
    args = parser.parse_args()
    main(args)
