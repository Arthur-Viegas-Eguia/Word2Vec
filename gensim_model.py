# -*- coding: utf-8 -*-

import gensim
import pandas as pd
import numpy as np

df = pd.Series(open("/content/answer.txt", "r").readlines())
print(df.shape)
vocab_tokens = df.apply(gensim.utils.simple_preprocess)
model = gensim.models.Word2Vec(window = 5, min_count = 1, workers = 2)
model.build_vocab(vocab_tokens, progress_per = 1000)
model.train(vocab_tokens, total_examples=model.corpus_count, epochs = 10)

model.save("model_gensim_discord.txt")

model.wv.most_similar("definitely")