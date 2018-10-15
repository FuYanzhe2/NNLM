#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools
curr_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(curr_path,'data')

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input.zh.txt")
        vocab_file = os.path.join(data_dir, "vocab.zh.pkl")

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            # print ("sent:", sent)
            # Update word_counts with each sent
            word_counts.update(sent)

        # word_counts.most_common():return the list of TOP N words of word_counts or ALL words if N is not refferred.
        # vocabulary_inv: ['<START>', '<UNK>', '<END>'] and  the words whose frq > self.mini_frq
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        # vocabulary: {word1:index1, wor2:index2, ...}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]

        # lines: each item of lines is a list of words of the initial line
        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        # print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        # self.vocab.get(w, 1) would return 1 if w is not the key of self.vocab
        raw_data = [[0] * self.seq_length +
            [self.vocab.get(w, 1) for w in line] +
            [2] * self.seq_length for line in lines]
        # each item of raw_data is a list of [0]*self.seq_length + the index of w in vocab and [2]*self.seq_length
        self.raw_data = raw_data

    def create_batches(self):
        i = 0
        xdata, ydata = list(), list()
        for row in self.raw_data:
            if i == 5:
                break
            i = i + 1
            # each item of raw_data matches a line of input file
            # len(row) = self.seq_length + len(line) + self.seq_length
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind-self.seq_length:ind])
                ydata.append([row[ind]])
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        # Convert xdata,ydata from list to np.array
        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


if __name__ == '__main__':
    textloader = TextLoader(data_dir = data_path, batch_size = 5, seq_length = 2, mini_frq=3)
