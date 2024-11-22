import re
import os
from vocabulary import Vocab
import numpy as np
from word2vec import Word2Vec
import most_similar
from multiprocessing import Pool
from functools import partial

def load_sample(idx, reviews, model):
    print(f"Sampling review {idx + 1}/{len(reviews)}...")
    samples = model.make_training_data(reviews[idx])
    return samples

def main():
    num_samples = 25000

    stopwords = set()
    with open('stopwords.txt') as f:
        for line in f:
            stopwords.add(line.strip('\n'))

    reviews = []
    for file in os.listdir('training_data'):
        with open(f'training_data/{file}') as f:
            review = re.sub('[^a-z| ]', '', ' '.join(f.readlines()).lower())
            while '  ' in review:
                review = review.replace('  ', ' ')
            reviews.append(review.split())

    reviews = reviews[:num_samples]

    v = Vocab(reviews, stopwords, min_count=10)

    for i in range(len(reviews)):
        reviews[i] = [v.index_from_word(word) for word in reviews[i] if word in v.vocab]

    model = Word2Vec(128, v, num_ns=7, window_size=5, learning_rate=.001)

    samples = None
    with Pool(8) as p:
        samples = p.map(partial(load_sample, reviews=reviews, model=model), range(num_samples))
    samples = np.concatenate(samples, axis=0)
        
    model.train(samples, 10)

    with open('embeds.txt', 'w') as f:
        embeds = model.get_target_embeddings()
        for key in embeds.keys():
            embeds[key] = embeds[key].tolist()
        f.write(str(embeds))
    with open('joint_embeds.txt', 'w') as f:
        embeds = model.get_joint_embeddings()
        for key in embeds.keys():
            embeds[key] = embeds[key].tolist()
        f.write(str(embeds))
    embeds = model.get_joint_embeddings()

    print(most_similar.most_similar_by_word('bad', embeds, 5))
    print(most_similar.most_similar_by_word('good', embeds, 5))


if __name__ == '__main__':
    main()
