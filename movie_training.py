import re
import os
from vocabulary import Vocab
import numpy as np
from word2vec import Word2Vec
import most_similar

num_samples = 2000

stopwords = set()
with open('stopwords.txt') as f:
    for line in f:
        stopwords.add(line.strip('\n'))

reviews = []
for file in os.listdir('training_data'):
    with open(f'training_data/{file}') as f:
        review = re.sub('[^(a-z| )]', '', ' '.join(f.readlines()).lower())
        while '  ' in review:
            review = review.replace('  ', ' ')
        reviews.append(review.split())

reviews = reviews[:num_samples]

v = Vocab(reviews, stopwords, min_count=10)

for i in range(len(reviews)):
    reviews[i] = [v.index_from_word(word) for word in reviews[i] if word in v.vocab]

model = Word2Vec(128, v, num_ns=7, window_size=5, learning_rate=.001)

samples = None
for i in range(num_samples):
    print(f"Sampling review {i + 1}/{num_samples}...")
    if samples is None:
        samples = model.make_training_data(reviews[i])
    else:
        samples = np.append(samples, model.make_training_data(reviews[i]), axis=0)
model.train(samples, 15)

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
