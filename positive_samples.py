import numpy as np

def make_positive_samples(corpus, window_size):
    skipgrams = []
    for i in range(len(corpus)):
        for j in range(1, window_size + 1):
            if i - j > -1:
                skipgrams.append((corpus[i], corpus[i - j]))
            if i < len(corpus) - j:
                skipgrams.append((corpus[i], corpus[i + j]))
    return np.array(skipgrams)


if __name__ == '__main__':
    sentence = "The wide road shimmered in the hot sun"
    tokens = sentence.split()
    print(len(make_positive_samples(tokens, 2)))

print("Hello World")