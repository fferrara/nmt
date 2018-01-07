import re
import torch
import unicodedata
from torch.autograd import Variable

from app.lang import Lang


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filter_pairs(pairs, max_length):
    good_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re "
    )

    def filter_pair(p):
        return len(p[0].split(' ')) < max_length and len(
            p[1].split(' ')) < max_length and \
               p[0].startswith(good_prefixes)

    return [pair for pair in pairs if filter_pair(pair)]


def read_lang(lang, max_length, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('resources/%s.txt' % lang).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang('eng')
        output_lang = Lang(lang)
    else:
        input_lang = Lang(lang)
        output_lang = Lang('eng')

    print(max_length)

    pairs = filter_pairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_sentence(pair[0])
        output_lang.index_sentence(pair[1])

    return input_lang, output_lang, pairs


def variable_from_sentence(lang, sentence, use_cuda=False):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(Lang.EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    #     print('var =', var)
    if use_cuda: var = var.cuda()
    return var


def variables_from_pair(input_lang, output_lang, pair, use_cuda=False):
    input_variable = variable_from_sentence(input_lang, pair[0], use_cuda)
    target_variable = variable_from_sentence(output_lang, pair[1], use_cuda)
    return input_variable, target_variable
