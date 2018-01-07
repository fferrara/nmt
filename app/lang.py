class Lang:
    SOS_token = 0
    EOS_token = 1

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.index_pointer = 2

    @property
    def n_words(self):
        return self.index_pointer

    def index_sentence(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
            return

        self.word2index[word] = self.index_pointer
        self.word2count[word] = 1
        self.index2word[self.index_pointer] = word
        self.index_pointer += 1