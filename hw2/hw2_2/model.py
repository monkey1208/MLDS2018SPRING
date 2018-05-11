import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import ipdb
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional) 

    def forward(self, input, lengths, hidden):
        batch_size = input.size(0)
        output = self.embedding(input)
        packed = pack_padded_sequence(output, lengths, batch_first=True)
        #outputs = self.bn(self.embedding(input))
        #outputs = outputs.view(batch_size, 80, -1)
        output, hidden = self.gru(packed, hidden)
        if self.bidirectional:
            # Sum bidirectional outputs
            output = (output[:, :self.hidden_size] + output[:, self.hidden_size:])
        return output, hidden
    def initHidden(self, num_layers, batch_size):
        if self.bidirectional:
            num_layers = num_layers * 2
        if torch.cuda.is_available():
            return torch.zeros(num_layers, batch_size, self.hidden_size).cuda()
        return torch.zeros(num_layers, batch_size, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, caption, lengths, hidden):
        output = self.embedding(caption)
        packed = pack_padded_sequence(output, lengths, batch_first=True)
        # output = F.relu(output)
        output, hidden = self.gru(packed, hidden)
        output = self.out(output[0])
        return output, hidden, packed[1]

class VanillaDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super(VanillaDecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, caption, hidden, encoder_outputs):
        batch_size = caption.size(0)
        output = self.embedding(caption)
        output = output.view(batch_size, 1, self.hidden_size)
        # output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = output.squeeze(1)
        #output = self.softmax(self.out(output))
        output = self.out(output)
        return output, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, max_len))  # B x S

        if torch.cuda.is_available():
            attn_energies = attn_energies.cuda()

        '''
        # For each batch of encoder outputs
        for b in range(batch_size):
        # Calculate energy for each encoder output
        for i in range(max_len):
        attn_energies[b, i] = self.score(
        hidden[b], encoder_outputs[b, i].unsqueeze(0))
        '''
        for b in range(batch_size):
            attn_energies[b] = self.score(
            hidden[b].unsqueeze(0), encoder_outputs[b])

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.mm(hidden, encoder_output.transpose(0, 1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.mm(hidden, energy.transpose(0, 1))
            return energy

        elif self.method == 'concat':
            hidden = hidden.repeat(encoder_output.size(0), 1)
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.mm(self.v, energy.transpose(0, 1))
            return energy

class BAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size,num_layers=1, dropout=0.1):
        super(BAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        batch_size = word_input.size(0)
        word_embedded = self.embedding(word_input)
        word_embedded = self.dropout(word_embedded)
        word_embedded = word_embedded.view(batch_size, 1, -1)  # B x 1 x N

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)  # B x 1 x N
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        rnn_input = F.relu(rnn_input)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(1)  # B x N
        output = self.out(output)

        # Return final output, hidden state
        return output, hidden
class LAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, method='general', num_layers=1, dropout=0.1):
        super(LAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.method = method
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        # Define layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

        self.attn = Attn(method, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(batch_size, 1, -1)  # B x 1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(1)  # B x S=1 x N -> B x N
        context = context.squeeze(1)        # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state
        return output, hidden
