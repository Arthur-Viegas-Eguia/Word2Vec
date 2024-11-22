import numpy as np
from vocabulary import Vocab
import positive_samples
import math
import most_similar

class Word2Vec:
    def __init__(self, dim_size, vocab: Vocab, unigram_frequency_weight = .75, num_ns = 2, window_size = 2):
        '''Initializes word2vec model with dim_size dimensions where each word has a normally distributed standard vector'''
        #Initialize word embeddings
        self.target_embeddings = np.random.uniform(-.1, .1, size=(len(vocab.vocab), dim_size))
        self.context_embeddings = np.random.uniform(-.1, .1, size=(len(vocab.vocab), dim_size))

        # the word frequency table used for negative sampling
        self.word_frequency = vocab.word_counts ** unigram_frequency_weight / np.sum(vocab.word_counts ** unigram_frequency_weight)

        # various things that must be set up.
        self.vocab = vocab
        self.dim_size = dim_size
        self.num_ns = num_ns
        self.window_size = window_size

    def probability(self, target_emb, context_emb):
        '''Computes the probability that word_1 and word_2 occur in context using the sigmoid of their dot product'''
        return 1 / (1 + np.exp(-np.dot(target_emb, context_emb)))

    def make_training_data(self, words):
        samples = []
        indices = np.arange(len(self.vocab.vocab))
        for i in range(len(words)):
            for j in range(max(0, i - self.window_size), min(len(words), i + self.window_size + 1)):
                if i == j:
                    continue
                sample = np.zeros((2 + self.num_ns,))
                sample[0] = words[i]
                sample[1] = words[j]
                sampled_words = set()
                for k in range(self.num_ns):
                    neg_samp = np.random.choice(indices, p=self.word_frequency)
                    while neg_samp in sampled_words or neg_samp == i:
                        neg_samp = np.random.choice(indices, p=self.word_frequency)
                    sample[2 + k] = neg_samp
                    sampled_words.add(neg_samp)
                samples.append(sample)
        return np.array(samples)
    
    def compute_loss(self, sample):
        loss = math.log(self.probability(sample[0], sample[1]))
        for i in range(2, len(sample)):
            loss += math.log(self.probability(sample[0], sample[i]))
        return -loss
    
    def gradient_descent(self, samples, learning_rate):
        for sample in samples:
            target_emb = self.target_embeddings[sample[0]]
            context_embs = self.context_embeddings[sample[1:]]

            # update for target embedding
            target_update = (self.probability(target_emb, context_embs[0]) - 1) * context_embs[0]
            for i in range(1, len(context_embs)):
                target_update += self.probability(target_emb, context_embs) * context_embs[i]
            self.target_embeddings[sample[0]] -= target_update * learning_rate

            # update for positive context embedding
            pos_context_update = (self.probability(target_emb, context_embs[0]) - 1) * target_emb
            self.context_embeddings[sample[1]] -= pos_context_update * learning_rate
            
            # update negative context embeddings
            for i in range(1, len(context_embs)):
                neg_context_update = self.probability(target_emb, context_embs[i]) * target_emb
                self.context_embeddings[sample[i + 1]] -= neg_context_update * learning_rate

    def get_embeddings(self):
        return {self.vocab.inv_vocab[i]: self.target_embeddings[i] + self.context_embeddings[i] for i in range(len(self.target_embeddings))}


if __name__ == '__main__':
    v = Vocab(['im', 'so', 'mad', 'right', 'now'])
    model = Word2Vec(5, v, window_size=2, num_ns=4)
    print(model.make_training_data([4, 1, 2, 5, 3, 6, 4, 3, 4]))
