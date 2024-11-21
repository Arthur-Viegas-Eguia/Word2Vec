import numpy as np
from numpy.linalg import norm

def cos_sim(embedding1, embedding2):
    cosine = np.dot(embedding1,embedding2)/(norm(embedding1)*norm(embedding2))
    return cosine