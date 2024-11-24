import gensim
import gensim.downloader
import ast
import sys
import os
import train_word2vec
import numpy as np
from vocabulary import Vocab

def main():
    if len(sys.argv) != 4:
        print('Usage: python3 compare_models.py <dataset_1:naive txt> <dataset_2:gensim txt> <input_data:txt or directory of txts>')
        sys.exit()
    model = None
    with open(sys.argv[1]) as f:
        model = ast.literal_eval(f.readline())
    model_1 = gensim.models.KeyedVectors(128)
    for word, vec in model.items():
        model_1.add_vector(word, np.array(vec))
    model_2 = gensim.models.KeyedVectors.load(sys.argv[2])
    comp_model = gensim.downloader.load('glove-wiki-gigaword-50')

    # read in all data
    docs = [] 
    if os.path.isfile(sys.argv[3]):
        docs.append(train_word2vec.read_file(sys.argv[3]))
    else:
        for file in os.listdir(sys.argv[3]):
            docs.append(train_word2vec.read_file(f'{sys.argv[3]}/{file}'))

    stopwords = set()
    with open('stopwords.txt') as f:
        for line in f:
            stopwords.add(line.strip('\n'))
    
    # use these words to ground comparison
    v = Vocab(docs, stopwords, min_count=10)

    frequencies = v.word_counts / np.sum(v.word_counts)

    model_1_score = 0
    model_2_score = 0
    words_checked = 0
    for word, idx in v.vocab.items():
        if word not in model_1 or word not in model_2.wv or word not in comp_model:
            continue
        comp_words = set(map(lambda pair: pair[0], comp_model.most_similar(word, topn=20)))
        words_1 = set(map(lambda pair: pair[0], model_1.most_similar(word, topn=20)))
        words_2 = set(map(lambda pair: pair[0], model_2.wv.most_similar(word, topn=20)))
        multiplier = frequencies[idx]
        model_1_score += len(comp_words.union(words_1)) * multiplier
        model_2_score += len(comp_words.union(words_2)) * multiplier
    print(f'Model 1\'s score was: {model_1_score}')
    print(f'Model 2\'s score was: {model_2_score}')


if __name__ == '__main__':
    main()
