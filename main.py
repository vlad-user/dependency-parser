import pickle
import argparse
import sys

import numpy as np

import dependency_parser as parser

argparser = argparse.ArgumentParser()
argparser.add_argument('--train',
					   action='store_true',
					   help="The flag specifies the mode: train or test.")
argparser.add_argument('--test',
					   action='store_true',
					   help="The flag specifies the mode: train or test.")
args = vars(argparser.parse_args())

def train(train_fname='train.labeled', 
		  fname_prefix='log_',
		  n_epochs=1,
		  log_interval=100):
	with open('train.labeled') as fo:
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
	with open(fname_prefix + 'dparser.pkl', 'rb') as fo:
		dp = pickle.load(fo)

	with open(test_fname) as fo:
		raw_data = fo.read()

	tokenized_sents = parser.raw_data2tokenized_sents(raw_data)
	accuracy = []
	step = 0
	for actual_sent in tokenized_sents:
		step += 1
		predicted_sent = dp.predict(actual_sent)
		accuracy.append(parser.compute_accuracy(actual_sent[1:],
												predicted_sent[1:]))

		if step % log_interval == 0:
			parser.print_log({'step': step, 'accuracy':np.mean(accuracy)})

	print('Accuracy for test dataset: ', np.mean(accuracy))
if __name__ == '__main__':
	
	if args['train'] and args['test']:
		raise ValueError('Both flags are true.')

	elif args['train']:
		train()
	elif args['test']:
		evaluate()
	else:
		print('Nothing to do.')
