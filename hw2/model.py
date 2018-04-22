import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import ipdb
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num__layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size * 80, 4096)
        #outputs = self.bn(self.embedding(input))
        #outputs = outputs.view(batch_size, 80, -1)
        embedded = self.bn(self.embedding(input)).view(batch_size, 80, -1)
        output = embedded
        output, hidden = self.gru(output)
        return output, hidden
    '''
    def initHidden(self):
        result = Variable(torch.zeros(1, 80, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result
            '''

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, caption, lengths, hidden):

        output = self.embedding(caption)
        packed = pack_padded_sequence(output, lengths, batch_first=True)
        #output = F.relu(output)

        output, hidden = self.gru(packed, hidden)
        output = self.softmax(self.out(output[0]))
        #output = self.out(output[0])
        return output, hidden, packed[1]
    '''
    def initHidden(self):
        result = Variable(torch.zeros(1, 80, self.hidden_size))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result
    '''