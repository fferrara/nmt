import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, use_cuda=False):
        super().__init__()

        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input_sentence, hidden):
        # self.embedding(input[t])
        input_length = len(input_sentence)
        embedded = self.embedding(input_sentence).view(input_length, 1, -1)
        # output[t], hidden[t] = self.gru(embedded, hidden[t-1]
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size, use_cuda=False):
        super().__init__()

        self.use_cuda = use_cuda
        self.method = method
        self.hidden_size = hidden_size

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(2*hidden_size, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        attention_energies = Variable(torch.zeros(seq_len))
        if self.use_cuda: attention_energies = attention_energies.cuda()

        for i in range(seq_len):
            attention_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attention_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            return hidden.dot(encoder_output)

        if self.method == 'general':
            energy = self.attn(encoder_output)
            return hidden.dot(energy)

        if self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            return self.other.dot(energy)

class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention("concat", hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, last_output_word, last_hidden, encoder_outputs):
        word_embedded = self.embedding(last_output_word).view(1, 1, -1)
        word_embedded = self.dropout(word_embedded)

        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        output = output.squeeze(0)  # B x N
        context = context.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return attention weights (for visualization)
        return output, hidden, attn_weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1,
                 dropout_p=0.1):
        super().__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers,
                          dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attention(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        #  apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden
        # state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for
        # visualization)
        return output, context, hidden, attn_weights






