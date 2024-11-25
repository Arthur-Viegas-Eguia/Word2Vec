import gensim

model_name = input("What is the name of your model?\n")
model = gensim.models.word2vec.Word2Vec.load(f'models/model_gensim_{model_name}.model')
query = input("What query do you want to give the model? (,q to quit)\n")
while query != ',q':
    print(f'The most similar words to {query} are: {model.wv.most_similar(query)}')
    query = input("What query do you want to give the model? (,q to quit)\n")
