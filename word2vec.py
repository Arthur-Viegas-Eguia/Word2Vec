import numpy as np
from vocabulary import Vocab

class Word2Vec:
    def __init__(self, dim_size, vocab: Vocab, unigram_frequency_weight = .75, num_ns = 10):
        '''Initializes word2vec model with dim_size dimensions where each word has a normally distributed standard vector'''
        #Initialize word embeddings
        self.embedding_dict = []
        for _ in range(len(vocab.vocab)):
            vec = np.random.normal(0, 1, size=dim_size)
            self.embedding_dict.append(vec / (np.sqrt(np.sum(vec ** 2))))
        self.embedding_dict = np.array(self.embedding_dict)

        # the word frequency table used for negative sampling
        self.word_frequency = vocab.word_counts ** unigram_frequency_weight / np.sum(vocab.word_counts ** unigram_frequency_weight)

        # various things that must be set up.
        self.vocab = vocab
        self.dim_size = dim_size
        self.num_ns = num_ns

    def probability(self, idx_1, idx_2):
        '''Computes the probability that word_1 and word_2 occur in context using the sigmoid of their dot product'''
        1/(1 + np.exp(-np.dot(self.embedding_dict[idx_1], self.embedding_dict[idx_2])))

    def make_negative_samples(self, target_idx):
        frq = self.word_frequency.copy()
        print(frq)
        frq[target_idx] = 0
        frq /= np.sum(frq)
        print(frq)
        return np.random.choice(np.arange(len(self.vocab.vocab)), size=self.num_ns, p=frq, replace=False)

    def index_from_word(self, word):
        return self.vocab.vocab[word]
    
    def word_from_index(self, idx):
        return self.vocab.inv_vocab[idx]


if __name__ == '__main__':
    data = 'hello hello hello the quick brown fox jumped over the lazy dog'.split()
    vocab = Vocab(data)
    print(vocab.vocab)
    print(vocab.inv_vocab)
    print(vocab.word_counts)
    model = Word2Vec(5, vocab, num_ns=2)
    idx = model.index_from_word('hello')
    print(idx, model.make_negative_samples(idx))