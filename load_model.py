import ast
import numpy as np
import most_similar

model = {}
with open('joint_embeds.txt') as f:
    model = ast.literal_eval(f.readline())
for key, val in model.items():
    model[key] = np.array(model[key])

query = input("What word do you want to look up? (,q to quit) \n")
while query != ',q':
    print(f'The most similar words to {query} are: {most_similar.most_similar_by_word(query, model, 10)}')
    query = input("What word do you want to look up? (,1) to quit \n")
    