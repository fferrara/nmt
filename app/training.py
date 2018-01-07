import random

import math

import time
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from app.lang import Lang

teacher_forcing_ratio = 0.5
clip = 5.0


def train(input_variable, target_variable,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length, use_cuda=False):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Encoding
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare decoding
    decoder_input = Variable(torch.LongTensor([[Lang.SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Start from last hidden state from encoder
    if use_cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    use_teacher_forcing = random.random() < teacher_forcing_ratio
    for di in range(target_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = \
            decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output[0], target_variable[di])

        if use_teacher_forcing:
            # Use the ground-truth target as the next input
            decoder_input = target_variable[di]  # Next target is next input
        else:
            # Use network's own prediction as the next input

            # Get most likely word index (highest value) from output
            top_value, top_index = decoder_output.data.topk(1)
            ni = top_index[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if use_cuda: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == Lang.EOS_token:
                break

    loss.backward()
    clip_grad_norm(encoder.parameters(), clip)
    clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))