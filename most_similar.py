# The goal here is to take an input word and create a list of the n most similar words (cos sim)

def most_similar(target, vocab, topn):
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
        topn_similar.append(all_sims[most_similar])
        del all_sims[most_similar]
    return topn_similar