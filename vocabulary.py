from collections import defaultdict as dd
import numpy as np

class Vocab:
    def __init__(self, words):
        '''initializes vocab with array of words. Creates vocab, a map from words to indices; inv_vocab, 
        a map from indices to words; and word_counts, a list where each index is the count of that word's appearances'''
        self.vocab = {}
        self.word_counts = []
        index = 0
        for word in words:
            if word not in self.vocab:
                self.vocab[word.lower()] = index
                index += 1
                self.word_counts.append(0)
            self.word_counts[self.vocab[word]] += 1
        self.word_counts = np.array(self.word_counts)
        self.inv_vocab = self.inverse_vocab()

    def inverse_vocab(self):
        return {index: word for word, index in self.vocab.items()}
