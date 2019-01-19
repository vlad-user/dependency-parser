import pickle
import argparse
import sys
import os
from time import time

import numpy as np

import dependency_parser.dependency_parser as parser



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


def evaluate(test_fname='test.labeled', fname_prefix='log_',
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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train',
                           action='store_true',
                           help="The flag specifies the mode: train or test.")
    argparser.add_argument('--test',
                           action='store_true',
                           help="The flag specifies the mode: train or test.")
    args = vars(argparser.parse_args())
    if args['train'] and args['test']:
        raise ValueError('Both flags are true.')

    elif args['train']:
        train()
    elif args['test']:
        evaluate()
    else:
        print('Nothing to do.')
