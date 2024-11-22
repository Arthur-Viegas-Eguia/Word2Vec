import re
import os
from vocabulary import Vocab
import numpy as np

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

v = Vocab(reviews, stopwords)

print(reviews[0])
for i in range(len(reviews)):
    reviews[i] = [v.index_from_word(word) for word in reviews[i] if word in v.vocab]
print([v.word_from_index(idx) for idx in reviews[0]])
