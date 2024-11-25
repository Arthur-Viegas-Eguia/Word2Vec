import nltk

import collections
def get_vocab_size(file):
  vocabulary = collections.defaultdict(int)
  for lines in file:
    line = lines.split(" ")
    for word in line:
      vocabulary[word] += 1
  return len(vocabulary)

print(get_vocab_size(open("clean_movies.txt", "r")))