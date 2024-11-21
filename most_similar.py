from cosine_similarity import cos_sim
# The goal here is to take an input word and create a list of the n most similar words (cos sim)

def most_similar_by_word(target, vocab, topn):
    #1.Find the embedding of word, then calculate the cosine similarity between it and every word in the vocab.
    #1b.As you go, make a dict of tuples, where index 0 is the word and index 1 is the cosine similarity to target.
    #2.Find the highest cos sim in the dict, then remove that sim's tuple from the dict and append it to a list. Repeat step 2 topn times.

    target_embed = vocab[target]
    all_sims = {}
    topn_similar = []
    del vocab[target]
    for word in vocab:
        all_sims[word] = cos_sim(target_embed, vocab[word])
    for x in range(topn):
        most_similar = max(all_sims, key=lambda key: all_sims[key])
        topn_similar.append(most_similar)
        del all_sims[most_similar]
    return topn_similar

def most_similar(positive, negative, vocab, topn):
    # expected that positive will be a list of words (strings), same with negatives. Vocab will be a dictionary with all words as keys
    # and their embeddings as values. topn is the size of the list of similar words.
    pos_sum = 0
    neg_sum = 0
    key_vectors = 0
    for vector in positive:
        pos_sum += vector
        key_vectors += 1
    for vector in negative:
        neg_sum += vector
        key_vectors += 1
    avg_vec = pos_sum - neg_sum / key_vectors
    all_sims = {}
    topn_similar = []
    for word in vocab:
        all_sims[word] = cos_sim(avg_vec, vocab[word])
    for x in range(topn):
        most_similar = max(all_sims, key=lambda key: all_sims[key])
        topn_similar.append(most_similar)
        del all_sims[most_similar]
    return topn_similar
    
def similarity(key1, key2):
    return cos_sim(key1, key2)

def distance(key1, key2):
    return 1 - cos_sim(key1, key2)

def similarities(key, keys_all, vocab):
    # key is a word string, keys_all is a list thereof. vocab is dictionary with all word strings as keys and their embeddings as values.
    similarity_list = []
    for item in keys_all:
        similarity_list.append(item, ": ", cos_sim(vocab[key], vocab[item]))
    return similarity_list

def closer_than(key1, key2, vocab):
    baseline = cos_sim(key1, key2)
    closer_list = []
    for word in vocab:
        if cos_sim(key1, word) > baseline:
            closer_list.append(word, ": ", cos_sim(key1, word))
