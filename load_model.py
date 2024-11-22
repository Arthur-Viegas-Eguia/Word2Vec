import ast
import numpy as np
import most_similar

model = {}
with open('joint_embeds.txt') as f:
    model = ast.literal_eval(f.readline())
for key, val in model.items():
    model[key] = np.array(model[key])

print(most_similar.most_similar_by_word('good', model, 5))