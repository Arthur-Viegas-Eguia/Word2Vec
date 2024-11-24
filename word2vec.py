import numpy as np
from vocabulary import Vocab
import math

class Word2Vec:
    def __init__(self, dim_size, vocab: Vocab, unigram_frequency_weight, num_ns, window_size, learning_rate, compute_loss):
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
        self.compute_loss = compute_loss

    def normalize_embeds(self):
        '''This sets the magnitude of all vectors to 1'''
        norms = np.linalg.norm(self.target_embeddings, axis=1, keepdims=True)
        self.target_embeddings /= norms
        norms = np.linalg.norm(self.context_embeddings, axis=1, keepdims=True)
        self.context_embeddings /= norms

    def probability(self, target_emb, context_emb):
        '''Computes the probability that word_1 and word_2 occur in context using the sigmoid of their dot product'''
        return 1 / (1 + np.exp(-np.dot(target_emb, context_emb)))

    def make_training_data(self, words):
        '''Generates training data for an input corpus represented by an array of word indices'''
        samples = []
        for i in range(len(words)):
            for j in range(max(0, i - self.window_size), min(len(words), i + self.window_size + 1)):
                if i == j:
                    continue
                sample = np.zeros((2 + self.num_ns,), dtype=int)
                sample[0] = words[i]
                sample[1] = words[j]
                samples.append(self.make_negative_sample(sample))
        return np.array(samples)

    def make_negative_sample(self, sample):
        '''Creates negative samples for an input sample. This samples from word_frequency. We allow all negative samples except the target word.'''
        indices = np.arange(len(self.vocab.vocab))
        sample[2:] = np.random.choice(indices, size=self.num_ns, p=self.word_frequency, replace=False)
        for i in range(2, len(sample)):
            while sample[i] == sample[0]:
                sample[i] = np.random.choice(indices, p=self.word_frequency)
                break
        return sample
    
    def compute_average_loss(self, samples):
        targets = self.context_embeddings[samples[:,0]]
        contexts = self.target_embeddings[samples[:,1:]]
        total_loss = 0
        for i in range(len(targets)):
            loss = math.log(self.probability(targets[i], contexts[i][0]))
            for j in range(1, len(contexts[i])):
                loss += math.log(1 - self.probability(targets[i], contexts[i][j]))
            total_loss -= loss
        return total_loss / len(samples)
    
    def train(self, samples, epochs):
        '''
        Passes over all samples epochs times, computing gradients along the way. Will print average losses if self.compute_loss is true.
        '''
        suffix = '' if not self.compute_loss else f' Average Loss: {self.compute_average_loss(samples)}'
        print(f'Starting Training.{suffix}')
        for i in range(1, epochs + 1):
            print(f'Starting Epoch {i}...')
            np.random.shuffle(samples) # shuffling is important so that we aren't always computing the same gradients in the same order
            self.normalize_embeds()
            self.gradient_descent(samples, self.learning_rate)
            suffix = '' if not self.compute_loss else f' Average Loss: {self.compute_average_loss(samples)}'
            print(f'Epoch {i} Complete.{suffix}')
    
    def gradient_descent(self, samples, learning_rate):
        '''Computes gradients for each word with respect for each sample'''
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
    v = Vocab([['im', 'so', 'mad', 'right', 'now']])
    model = Word2Vec(5, v, window_size=2, num_ns=4)
    print(model.make_training_data([4, 1, 2, 5, 3, 6, 4, 3, 4]))
