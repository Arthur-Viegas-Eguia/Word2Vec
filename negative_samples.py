import numpy as np
import random

def make_negative_samples(corpus, window size, num_ns):
    skipgrams = []
    for i in range(len(corpus)):
        pos_words = []
        neg_words = []
        for j in range(1, window size + 1):
            if i - j > -1:
                pos_words.append(corpus[i - j])
            if i < len(corpus) - j:
                pos_words.append(corpus[i + j])
        for x in range(num_ns):
            rand_idx = random.rand_int(0, len(corpus) -1)
            if corpus[rand_idx] not in pos_words and corpus[rand_idx] not in neg_words:
                neg_words.append(corpus[rand_idx])
                skipgrams.append(corpus[i], corpus[rand_idx])
            else:
                while corpus[rand_idx] in pos_words or corpus[rand_idx] in neg_words:
                    rand_idx = random.rand_int(0, len(corpus) -1)
                neg_words.append(corpus[rand_idx])
                skipgrams.append(corpus[i], corpus[rand_idx])                
    return np.array(skipgrams)


if __name__ == '__main__':
    sentence = "The wide road shimmered in the hot sun"
    tokens = sentence.split()
    print(len(make_negative_samples(tokens, 2)))