import argparse
import random

import time
from torch import optim, nn

from app.input import read_lang, variables_from_pair
from app.model import Encoder, AttnDecoderRNN
from app.training import train, time_since

parser = argparse.ArgumentParser()
parser.add_argument('gpu', metavar='gpu', type=bool, nargs='?', default=False, help='whether we have GPU')

args = parser.parse_args()
if args.gpu:
    from settings.gpu import USE_CUDA, MAX_LENGTH
else:
    from settings.local import USE_CUDA, MAX_LENGTH

attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

input_lang, output_lang, pairs = read_lang('fra', MAX_LENGTH)
# Print an example pair
print(random.choice(pairs))

encoder = Encoder(input_lang.n_words, hidden_size, n_layers, USE_CUDA)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p, USE_CUDA)

# Move models to GPU
if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Configuring training
n_epochs = 50000
plot_every = 200
print_every = 1

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

for epoch in range(1, n_epochs + 1):
    # Get training data for this cycle
    training_pair = variables_from_pair(input_lang, output_lang, random.choice(pairs), USE_CUDA)
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    loss = train(input_variable, target_variable, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH, USE_CUDA)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (
        time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0