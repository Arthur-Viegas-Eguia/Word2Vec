import numpy as np
from vocabulary import Vocab

class Word2Vec:
    def __init__(self, dim_size, vocab: Vocab, unigram_frequency_weight = .75, num_ns = 10):
        '''Initializes word2vec model with dim_size dimensions where each word has a normally distributed standard vector'''
        #Initialize word embeddings
        self.embedding_dict = []
        for _ in range(len(vocab.vocab)):
            vec = np.random.normal(0, 1, size=dim_size)
            self.append(vec / (np.sqrt(np.sum(vec ** 2))))
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
        temp = self.word_frequency[target_idx]
        self.word_frequency[target_idx] = 0
        np.random.choice(np.arange(len(self.vocab.vocab)), size=self.num_ns, p=self.word_frequency)
        self.word_frequency[target_idx] = temp



if __name__ == '__main__':
    model = Word2Vec(5, ['hello'])
    print(model.embedding_dict['hello'])