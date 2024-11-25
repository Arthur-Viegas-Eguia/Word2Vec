import ast
import numpy as np
import most_similar

if __name__ == '__main__':
    name = input("What is the name of the model you'd like to open?\n")
    model = {}
    with open(f'models/{name}_embeds.txt') as f:
        model = ast.literal_eval(f.readline())
    for key, val in model.items():
        model[key] = np.array(model[key])
    print('Model loaded.')

    # handle analogy mode and query mode
    mode = input("Do you want to enter analogy or query mode? (a/q or ,q to quit)\n")
    while mode != ',q':
        if mode == 'a':
            query = input("___ is to ___ as ___ is to ___ (enter three words separated by spaces or ,q to quit)\n")
            while query != ',q':
                query = query.split()
                try:
                    ans = most_similar.most_similar_by_embedding(model[query[2]] + (model[query[1]] - model[query[0]]), model, 10)
                    print(f'{query[0]} is to {query[1]} as {query[2]} is to {ans}')
                except IndexError:
                    # less than three arguments given
                    print('Query should have three words separated by spaces')
                except KeyError:
                    # argument not in corpus
                    print('At least one word was not recognized.')
                query = input("___ is to ___ as ___ is to ___ (enter three words separated by spaces or ,q to quit)\n")
        else:
            query = input("What word do you want to look up? (,q to quit) \n")
            while query != ',q':
                try:
                    print(f'The most similar words to {query} are: {most_similar.most_similar_by_word(query, model, 10)}')
                except KeyError:
                    # word not in corpus
                    print(f'Query {query} was not found in the model. Please try another query.')
                query = input("What word do you want to look up? (,q to quit) \n")
        mode = input("Do you want to enter analogy or query mode? (a/q or ,q to quit)\n ")
