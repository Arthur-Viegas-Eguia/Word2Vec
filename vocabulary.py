from collections import defaultdict as dd
import numpy as np

class Vocab:
    def __init__(self, docs, stopwords=[], min_count=0):
        '''initializes vocab with a list of documents, each of which is represented by an array of words. Creates vocab, a map from words to indices; inv_vocab, 
        a map from indices to words; and word_counts, a list where each index is the count of that word's appearances'''
        self.vocab = {}
        self.word_counts = []
        index = 0
        for words in docs:
            for word in words:
                if word in stopwords:
                    continue
                if word not in self.vocab:
                    self.vocab[word.lower()] = index
                    index += 1
                    self.word_counts.append(0)
                self.word_counts[self.vocab[word]] += 1
        
        words_to_delete = []
        for key, val in self.vocab.items():
            if self.word_counts[val] < min_count:
                words_to_delete.append(key)

        self.inv_vocab = self.inverse_vocab()
        for i in range(len(words_to_delete)):
            idx = self.vocab[words_to_delete[i]]
            self.vocab[self.inv_vocab[len(self.vocab) - 1]] = idx
            self.inv_vocab[idx] = self.inv_vocab[len(self.vocab) - 1]
            self.word_counts[idx] = self.word_counts[-1]
            del self.vocab[words_to_delete[i]]
            del self.inv_vocab[len(self.inv_vocab) - 1]
            self.word_counts.pop(-1)
        
        self.word_counts = np.array(self.word_counts)

    def inverse_vocab(self):
        return {index: word for word, index in self.vocab.items()}
    
    def index_from_word(self, word):
        return self.vocab[word]
    
    def word_from_index(self, idx):
        return self.inv_vocab[idx]


if __name__ == '__main__':
    v = Vocab([['hello', 'hell', 'goodbye', 'good', 'hello', 'goodbye']], min_count=2)
    print(v.vocab)
    print(v.inv_vocab)
    print(v.word_counts)
