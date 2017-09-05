import torch
import numpy as np
import os

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if not word in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def check_word(self, word):
        exist = False
        if word in self.idx2word:
            exist = True
        return exist

    def get_index(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        ids = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if not self.dictionary.check_word(word):
                        self.dictionary.add_word(word)
                    ids.append(self.dictionary.get_index(word))
        return torch.from_numpy(np.asarray(ids))
