import gensim
import gensim.downloader
import ast
import sys
import os
import train_word2vec
import numpy as np
from vocabulary import Vocab
import argparse

def load_naive(file, dims):
    '''Loads a naive model from the file "file" with dimension size dims.'''
    data = None
    with open(file) as f:
        data = ast.literal_eval(f.readline())
    model = gensim.models.KeyedVectors(dims)
    for word, vec in data.items():
        model.add_vector(word, np.array(vec))
    return model


def load_gensim(file):
    '''Loads a gensim model from the file "file"'''
    return gensim.models.word2vec.Word2Vec.load(file).wv


def main():
    parser = argparse.ArgumentParser(
        prog='Word2Vec Model Comparison',
        description='Compare a model made using our naive implementation with a Gensim model.',
    )
    # get program parameters
    parser.add_argument('model_1', help='Directory to naive or gensim model (.txt or .model).')
    parser.add_argument('model_2', help='Directory to naive or gensim model (.txt or .model)')
    parser.add_argument('training_data', help='The directory of txt files or the txt file itself that should be used for training.')
    parser.add_argument('-d', '--dimensions', default=128, help='The number of dimensions for each vector.', type=int)
    parser.add_argument('-m', '--min_word_count', default=10, help='Minimum number of occurences for a word to be considered.', type=int)

    args = parser.parse_args()

    model_1 = load_naive(args.model_1, args.dimensions) if args.model_1.split('.')[-1] == 'txt' else load_gensim(args.model_1)
    model_2 = load_naive(args.model_2, args.dimensions) if args.model_2.split('.')[-1] == 'txt' else load_gensim(args.model_2)
    comp_model = gensim.downloader.load('glove-wiki-gigaword-50')

    # read in all data
    docs = [] 
    if os.path.isfile(args.training_data):
        docs.append(train_word2vec.read_file(args.training_data))
    else:
        for file in os.listdir(args.training_data):
            docs.append(train_word2vec.read_file(f'{args.training_data}/{file}'))

    stopwords = set()
    with open('stopwords.txt') as f:
        for line in f:
            stopwords.add(line.strip('\n'))
    
    # use these words to ground comparison, these will be the words in the naive model and we only compare words that are in all three, so this works.
    v = Vocab(docs, stopwords, min_count=args.min_word_count)

    frequencies = v.word_counts / np.sum(v.word_counts)

    # compare
    model_1_score = 0
    model_2_score = 0
    for word, idx in v.vocab.items():
        if word not in model_1 or word not in model_2 or word not in comp_model:
            continue
        comp_words = set(map(lambda pair: pair[0], comp_model.most_similar(word, topn=20)))
        words_1 = set(map(lambda pair: pair[0], model_1.most_similar(word, topn=10)))
        words_2 = set(map(lambda pair: pair[0], model_2.most_similar(word, topn=10)))
        multiplier = frequencies[idx]
        model_1_score += len(comp_words.intersection(words_1)) * multiplier
        model_2_score += len(comp_words.intersection(words_2)) * multiplier
    print(f'Naive\'s score was: {model_1_score}')
    print(f'Gensim\'s score was: {model_2_score}')


if __name__ == '__main__':
    main()
