import pickle
import argparse
import sys
import os
from time import time

import numpy as np

import dependency_parser.dependency_parser as parser
edge2rawfeature10 = parser.edge2rawfeature10

def train(train_fname='train.labeled', 
          fname_prefix='log_',
          n_epochs=100,
          log_interval=1000):
    path = os.path.join(os.path.dirname(__file__), 'dependency_parser', 'data')
    train_fname = os.path.join(path, train_fname)
    
    with open(train_fname) as fo:
        raw_data = fo.read()
    
    dp = parser.DependencyParser(raw_data)
    tokenized_sents = parser.raw_data2tokenized_sents(raw_data)
    data = dp.prepare_train_data(tokenized_sents)
    dp.optimize(data,
                n_epochs=n_epochs,
                log_interval=log_interval,
                fname_prefix=fname_prefix)
    with open(fname_prefix + 'dparser.pkl', 'wb') as fo:
        pickle.dump(dp, fo, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(test_fname='test.labeled', fname_prefix='m1_50_',
             log_interval=10):
    """TODO"""
    path = os.path.join(os.path.dirname(__file__), 'dependency_parser', 'data')
    test_fname = os.path.join(path, test_fname)
    with open(fname_prefix + 'dparser.pkl', 'rb') as fo:
        dp = pickle.load(fo)

    with open(test_fname) as fo:
        raw_data = fo.read()

    tokenized_sents = parser.raw_data2tokenized_sents(raw_data)
    accuracy = []
    times = []
    step = 0
    for actual_sent in tokenized_sents:
        start_time = time()
        step += 1
        predicted_sent = dp.predict(actual_sent)
        accuracy.append(parser.compute_accuracy(actual_sent[1:],
                                                predicted_sent[1:]))
        times.append(time() - start_time)

        if step % log_interval == 0:
            parser.print_log({'step': step,
                              'accuracy':"{0:.4f}".format(np.mean(accuracy)),
                              'time':"{0:.4f}".format(np.mean(times)) + ' sec/sent'})
    print()
    print('Accuracy for test dataset: ', "{0:.4f}".format(np.mean(accuracy)))
    return np.mean(accuracy)

def label_unlabeled(dp_fname, fname2label, outfname):
    with open(dp_fname, 'rb') as fo:
        dp = pickle.load(fo)

    with open(fname2label) as fo:
        raw_data = fo.read()

    splitted_sents = raw_data.split('\n\n')
    raw_sents = [x.split('\n') for x in splitted_sents]
    tokenized_sents = parser.raw_data2tokenized_sents(raw_data)
    result = []

    for raw_sent, tokenized_sent in zip(raw_sents, tokenized_sents):
        predicted_sent = dp.predict(tokenized_sent)
        words = []
        for raw_word, pred_word in zip(raw_sent, predicted_sent[1:]):
            splitted_word = raw_word.split('\t')
            try:
                splitted_word[6] = str(pred_word[3])
            except IndexError:
                print(splitted_word)
                print(pred_word)
                raise
            words.append('\t'.join(splitted_word))
        result.append('\n'.join(words))
    
    with open(outfname, 'w') as fo:
        fo.write('\n\n'.join(result))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train',
                           action='store_true',
                           help="The flag specifies the mode: train or test.")
    argparser.add_argument('--test',
                           action='store_true',
                           help="The flag specifies the mode: train or test.")
    argparser.add_argument('--prefix',
                           required=True,
                           type=str,
                           help="Model's prefix string.")
    args = vars(argparser.parse_args())
    if args['train'] and args['test']:
        raise ValueError('Both flags are true.')

    elif args['train']:
        train(fname_prefix=args['prefix'])
    elif args['test']:
        evaluate(fname_prefix=args['prefix'])
    else:
        print('Nothing to do.')
