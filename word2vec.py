import numpy as np
from vocabulary import Vocab
import positive_samples
import math
import most_similar

class Word2Vec:
    def __init__(self, dim_size, vocab: Vocab, unigram_frequency_weight = .75, num_ns = 2, window_size = 2, learning_rate = .025, norm_update_frq = 100):
        '''Initializes word2vec model with dim_size dimensions where each word has a normally distributed standard vector'''
        #Initialize word embeddings
        self.target_embeddings = np.random.uniform(-.01, .01, size=(len(vocab.vocab), dim_size))
        self.context_embeddings = np.random.uniform(-.01, .01, size=(len(vocab.vocab), dim_size))

        # the word frequency table used for negative sampling
        self.word_frequency = (vocab.word_counts ** unigram_frequency_weight) / np.sum(vocab.word_counts ** unigram_frequency_weight)

        # various things that must be set up.
        self.vocab = vocab
        self.dim_size = dim_size
        self.num_ns = num_ns
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.norm_update_frq = norm_update_frq

    def normalize_embeds(self):
        norms = np.linalg.norm(self.target_embeddings, axis=1, keepdims=True)
        self.target_embeddings /= norms
        norms = np.linalg.norm(self.context_embeddings, axis=1, keepdims=True)
        self.context_embeddings /= norms

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
                sample = np.zeros((2 + self.num_ns,), dtype=int)
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
        loss = math.log(self.probability(self.target_embeddings[sample[0]], self.context_embeddings[sample[1]]))
        for i in range(2, len(sample)):
            loss += math.log(1 - self.probability(self.target_embeddings[sample[0]], self.context_embeddings[sample[i]]))
        return -loss
    
    def train(self, samples, epochs):
        for i in range(epochs):
            print(f'Starting Epoch {i}...')
            # normalizing embeddings helps ensure that vectors don't get too big. This doesn't happen after the last epoch so some magnitude can be accounted for
            self.normalize_embeds()
            self.gradient_descent(samples, self.learning_rate)
            total_loss = 0
            for sample in samples:
                total_loss += self.compute_loss(sample)
            print(f'Epoch {i} Complete. Average Loss: {total_loss / len(samples)}.')
    
    def gradient_descent(self, samples, learning_rate):
        for sample in samples:
            target_emb = self.target_embeddings[sample[0]]
            context_embs = self.context_embeddings[sample[1:]]

            # update for target embedding
            target_update = (self.probability(target_emb, context_embs[0]) - 1) * context_embs[0]
            for j in range(1, len(context_embs)):
                target_update += self.probability(target_emb, context_embs[j]) * context_embs[j]
            self.target_embeddings[sample[0]] -= target_update * learning_rate

            # update for positive context embedding
            pos_context_update = (self.probability(target_emb, context_embs[0]) - 1) * target_emb
            self.context_embeddings[sample[1]] -= pos_context_update * learning_rate
            
            # update negative context embeddings
            for j in range(1, len(context_embs)):
                neg_context_update = self.probability(target_emb, context_embs[j]) * target_emb
                self.context_embeddings[sample[j + 1]] -= neg_context_update * learning_rate

    def get_target_embeddings(self):
        return {self.vocab.inv_vocab[i]: self.target_embeddings[i] for i in range(len(self.target_embeddings))}
    
    def get_context_embeddings(self):
        return {self.vocab.inv_vocab[i]: self.context_embeddings[i] for i in range(len(self.target_embeddings))}

    def get_joint_embeddings(self):
        return {self.vocab.inv_vocab[i]: self.target_embeddings[i] + self.context_embeddings[i] for i in range(len(self.target_embeddings))}


if __name__ == '__main__':
    v = Vocab(['im', 'so', 'mad', 'right', 'now'])
    model = Word2Vec(5, v, window_size=2, num_ns=4)
    print(model.make_training_data([4, 1, 2, 5, 3, 6, 4, 3, 4]))
