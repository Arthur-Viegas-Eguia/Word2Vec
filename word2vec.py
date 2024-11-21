import numpy as np
from vocabulary import Vocab
import positive_samples
import math
import most_similar

class Word2Vec:
    def __init__(self, dim_size, vocab: Vocab, unigram_frequency_weight = .75, num_ns = 10):
        '''Initializes word2vec model with dim_size dimensions where each word has a normally distributed standard vector'''
        #Initialize word embeddings
        self.target_embeddings = []
        self.context_embeddings = []
        for _ in range(len(vocab.vocab)):
            target_vec = np.random.normal(0, 1, size=dim_size)
            context_vec = np.random.normal(0, 1, size=dim_size)
            self.target_embeddings.append(target_vec / (np.sqrt(np.sum(target_vec ** 2))))
            self.context_embeddings.append(context_vec / (np.sqrt(np.sum(context_vec ** 2))))
        self.target_embeddings = np.array(self.target_embeddings)
        self.context_embeddings = np.array(self.context_embeddings)

        # the word frequency table used for negative sampling
        self.word_frequency = vocab.word_counts ** unigram_frequency_weight / np.sum(vocab.word_counts ** unigram_frequency_weight)

        # various things that must be set up.
        self.vocab = vocab
        self.dim_size = dim_size
        self.num_ns = num_ns

    def probability(self, target_word, context_word):
        '''Computes the probability that word_1 and word_2 occur in context using the sigmoid of their dot product'''
        return 1 / (1 + np.exp(-np.dot(self.target_embeddings[target_word], self.context_embeddings[context_word])))

    def make_training_data(self, words, window_size):
        skipgrams = positive_samples.make_positive_samples(words, window_size)
        return self.make_negative_samples(skipgrams)


    def make_negative_samples(self, skipgrams):
        for i in range(len(skipgrams)):
            frq = self.word_frequency.copy()
            frq[skipgrams[i][0]] = 0
            frq /= np.sum(frq)
            skipgrams[i] = np.append(skipgrams[i], np.random.choice(np.arange(len(self.vocab.vocab)), size=self.num_ns, p=frq, replace=False))
        return np.array(skipgrams)
            

    def index_from_word(self, word):
        return self.vocab.vocab[word]
    
    def word_from_index(self, idx):
        return self.vocab.inv_vocab[idx]
    
    def compute_loss(self, sample):
        loss = math.log(self.probability(sample[0], sample[1]))
        for i in range(2, len(sample)):
            loss += math.log(self.probability(sample[0], sample[i]))
        return -loss
    
    def gradient_descent(self, samples, learning_rate):
        target_updates = np.zeros(shape=(self.context_embeddings.shape))
        target_update_counts = np.zeros(shape=len(self.context_embeddings))
        context_updates = np.zeros(shape=(self.context_embeddings.shape))
        context_update_counts = np.zeros(shape=len(self.context_embeddings))
        for sample in samples:
            target_update = (self.probability(sample[0], sample[1]) - 1) * self.context_embeddings[sample[1]]
            for i in range(2, len(sample)):
                target_update += self.probability(sample[0], sample[i]) * self.context_embeddings[sample[i]]
            target_updates[sample[0]] += target_update
            target_update_counts[sample[0]] += 1
            pos_context_update = (self.probability(sample[0], sample[1]) - 1) * self.target_embeddings[sample[0]]
            context_updates[sample[1]] += pos_context_update
            context_update_counts[sample[1]] += 1
            for i in range(2, len(sample)):
                neg_context_update = self.probability(sample[0], sample[i]) * self.target_embeddings[sample[0]]
                context_updates[sample[i]] += neg_context_update
                context_update_counts[sample[i]] += 1
        for i in range(len(target_updates)):
            self.target_embeddings[i] += (target_updates[i] / target_update_counts[i]) * learning_rate
            self.context_embeddings[i] += (context_update_counts[i] / context_update_counts[i]) * learning_rate
        self.target_embeddings = self.target_embeddings / np.sqrt(np.sum(self.target_embeddings ** 2, axis=0))
        self.context_embeddings = self.context_embeddings / np.sqrt(np.sum(self.context_embeddings ** 2, axis=0))

    def get_embeddings(self):
        return {vocab.inv_vocab[i]: self.target_embeddings[i] + self.context_embeddings[i] for i in range(len(self.target_embeddings))}


if __name__ == '__main__':
    data = None
    with open('modified_output.txt') as f:
        data = f.readline().lower()
    data = data.split()
    vocab = Vocab(data)
    sentence = [vocab.vocab[word] for word in data]
    print('Initializing Model...')
    model = Word2Vec(128, vocab, num_ns=2)
    print('Making Samples...')
    samples = model.make_training_data(sentence, 2)
    print('Computing Starting Loss...')
    total_loss = 0
    for sample in samples:
        total_loss += model.compute_loss(sample)
    print(total_loss / len(samples))
    for i in range(5):
        print(f'Starting Epoch {i}...')
        model.gradient_descent(samples, .1)
    print('Computing Loss...')
    total_loss = 0
    for sample in samples:
        total_loss += model.compute_loss(sample)
    print(total_loss / len(samples))
    embeddings = model.get_embeddings()
    print(most_similar.most_similar('bad', embeddings, 2))
