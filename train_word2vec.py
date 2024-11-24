import argparse
import os
import re
from vocabulary import Vocab
from word2vec import Word2Vec
import numpy as np
from multiprocessing import Pool
from functools import partial
import sys

def read_file(filename):
    '''reads a file, stripping all non letter characters and splitting each word into its own entity of an array'''
    content = None
    with open(filename) as f:
        content = re.sub("'", '', ' '.join(f.readlines()).lower()) # This will get rid of all ' because contractions should not be split into two words. Like can't should not become [can, t]
        content = re.sub('[^a-z]', ' ', content) # this goes through all non-letter characters and puts a space where they were so danger-seeking becomes danger seeking for example
        while '  ' in content:
            content = content.replace('  ', ' ') # we don't want double spaces
    return content.split()


def prepare_sample(doc, model):
    '''This readies a document to be passed into the model. It takes in a document and makes all samples for that document'''
    sample = model.make_training_data(doc)
    return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Word2Vec',
        description='Train your own word2vec model on a corpus of your choice! To change stopwords, please edit stopwords.txt in the same directory as this file.',
        epilog='With this, training word2vec has never been easier! (this is definitely not true, but we\'re going to roll with it.)'
    )
    # get program parameters
    parser.add_argument('model_name', help='The name the model will be given in storage.')
    parser.add_argument('file_or_directory', help='The directory of txt files or the txt file itself that should be used for training.')
    parser.add_argument('-d', '--dimensions', default=128, help='The number of dimensions for each vector.', type=int)
    parser.add_argument('-e', '--epochs', default=10, help='Number of iterations over the dataset while training.', type=int)
    parser.add_argument('-w', '--window_size', default=5, help='Number of surrounding words to consider as context.', type=int)
    parser.add_argument('-n', '--negative_samples', default=7, help='The number of negative samples for each training sample.', type=int)
    parser.add_argument('-m', '--min_word_count', default=10, help='Minimum number of occurences for a word to be considered.', type=int)
    parser.add_argument('-u', '--unigram_weight', default=.75, help='Normalizing factor to increase probability of rare words being negative sample (1 for no reweight).', type=float)
    parser.add_argument('-l', '--learning_rate', default=.001, help='Learning rate for each epoch.', type=float)
    parser.add_argument('-c', '--compute_loss', action='store_true', help='Whether to print out loss after each epoch. Significantly raises computation time and may or may not be meaningful.')

    args = parser.parse_args()
    
    # check that training data exists
    if not os.path.exists(args.file_or_directory):
        print('Could not find file/directory.')

    # make the folder models if it doesn't exist already
    try:
        os.mkdir('models')
    except FileExistsError:
        pass

    # prevent people from accidentally deleting old models
    if os.path.exists(f'models/{args.model_name}_embeds.txt') or os.path.exists(f'models/{args.model_name}_joint_embeds.txt'): 
        response = input(f'You already have a model named {args.model_name}. Running this program will delete it. Do you want to proceed? (y/n)\n')
        if response.lower() != 'y':
            sys.exit()
    
    # if the name is somehow illegal in the operating system or something, I want to tell users before they wait for it to train
    with open(f'models/{args.model_name}_embeds.txt', 'w'):
        pass
    with open(f'models/{args.model_name}_joint_embeds.txt', 'w'):
        pass

    # read in all data
    docs = [] 
    if os.path.isfile(args.file_or_directory):
        docs.append(read_file(args.file_or_directory))
    else:
        for file in os.listdir(args.file_or_directory):
            docs.append(read_file(f'{args.file_or_directory}/{file}'))

    # collect stopwords
    stopwords = set()
    with open('stopwords.txt') as f:
        for line in f:
            stopwords.add(line.strip('\n'))

    # initialize vocab and model with parameters from arguments
    v = Vocab(docs, stopwords, min_count=args.min_word_count)
    model = Word2Vec(args.dimensions, v, unigram_frequency_weight= args.unigram_weight, num_ns=args.negative_samples, window_size=args.window_size, learning_rate=args.learning_rate, compute_loss=args.compute_loss)

    # turn all documents from words into indices, numbers are faster than strings.
    for i in range(len(docs)):
        docs[i] = [v.index_from_word(word) for word in docs[i] if word in v.vocab]

    # prepare samples using 8 CPU processes to make it as efficient as possible. Only is faster for folder reading, not individual file.
    with Pool(8) as p:
        print('Reading Samples. This may take some time...')
        samples = p.map(partial(prepare_sample, model=model), docs)
    samples = np.concatenate(samples, axis=0)
    print('Samples loaded.')

    # where the magic happens, the model trains.
    model.train(samples, args.epochs)

    # save results so we can use them.
    with open(f'models/{args.model_name}_embeds.txt', 'w') as f:
        embeds = model.get_target_embeddings()
        for key in embeds.keys():
            embeds[key] = embeds[key].tolist()
        f.write(str(embeds))
    with open(f'models/{args.model_name}_joint_embeds.txt', 'w') as f:
        embeds = model.get_joint_embeddings()
        for key in embeds.keys():
            embeds[key] = embeds[key].tolist()
        f.write(str(embeds))
        