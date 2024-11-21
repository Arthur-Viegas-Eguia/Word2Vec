def index_words(words):
    vocab = {'<pad>': 1}
    index = 2
    for word in words:
        if word not in vocab:
            vocab[word.lower()] = index
            index += 1
    return vocab

def inverse_vocab(vocab):
    return {index: word for word, index in vocab.items()}
