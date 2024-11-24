import ast
import numpy as np
import most_similar

if __name__ == '__main__':
    name = input("What is the name of the model you'd like to open?\n")
    model = {}
    with open(f'models/{name}_joint_embeds.txt') as f:
        model = ast.literal_eval(f.readline())
    for key, val in model.items():
        model[key] = np.array(model[key])

    print('Model loaded.')
    query = input("What word do you want to look up? (,q to quit) \n")
    while query != ',q':
        try:
            print(f'The most similar words to {query} are: {most_similar.most_similar_by_word(query, model, 10)}')
            query = input("What word do you want to look up? (,q) to quit \n")
        except KeyError:
            print(f'Query {query} was not found in the model. Please try another query.')
            query = input("What word do you want to look up? (,q) to quit \n")
    
    print(most_similar.most_similar_by_embedding(model['woman'] + (model['actor'] - model['man']), model, 10))