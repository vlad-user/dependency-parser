import sys
import collections
import itertools
import pickle
import random
import gc
import os

import numpy as np

from dependency_parser.chu_liu import Digraph

ROOT_TOKEN = (0, '<ROOT>', '<ROOT-POS>')

def compute_accuracy(y_true, y_pred):

    result = [y1 == y2 for y1, y2 in zip(y_true, y_pred)]
    return sum(result) / len(result)

def raw_data2tokenized_sents(raw_data):
    """Converts raw inputs data to a list of lists of 4-tuples.

    Args:
        raw_data: A sentences splitable by `\n\n` character.
            For example, the sentence must have following form:
            `'1\t@\t_\tIN\t_\t_\t0\tROOT\t_\t_'`

    Returns:
        A list of list of 4-tuples.
    """
    raw_sents = []
    for raw_words in raw_data.split('\n\n')[:-1]:
        sent = [ROOT_TOKEN + (-1, )]

        splitted_words = [w.split('\t') for w in raw_words.split('\n')]

        for w in splitted_words:
            try:
                sent.append((int(w[0]), w[1], w[3], int(w[6])))
            except:
                sent.append((int(w[0]), w[1], w[3]))
        raw_sents.append(sent)
    return raw_sents

def tokenized_sent2rawgraph(tokenized_sent):
    """Creates raw graph from tokenized sentence.

    Given input `tokenized_sent` as python list, e.g.
    `tokenized_sent = [(0, 'ROOT', 'ROOT_POS', -1), (1, 'Hello', 'NN', 0)...]`
    generates a graph s.t.
    `graph = {(0, 'ROOT', 'ROOT_POS'): [(1, 'Hello', 'NN'), ...],
              (i, some-parent, some-parent-pos): [list of all children]...}`

    """

    graph = {tokenized_sent[i][:3]: [] for i in range(len(tokenized_sent))}

    for parent in tokenized_sent:
        children = [t for t in tokenized_sent if t[3] == parent[0]]
        for child in children:
            graph[parent[:3]].append(child[:3])

    return graph

def tokenized_sent2rawclique(tokenized_sent):
    """Converts `tokenized_sent` to a clique graph."""
    graph = {tokenized_sent[i][:3]: [] for i in range(len(tokenized_sent))}

    for i in range(len(tokenized_sent)):
        for j in range(len(tokenized_sent)):
            if j != i and j != 0:
                graph[tokenized_sent[i][:3]].append(tokenized_sent[j][:3])

    return graph

def graph2edges(graph):
    """Converts graph dictionary into the edge-children representation."""
    edges = [(src, dst) for src, dsts in graph.items() for dst in dsts]
    return edges

def rawgraph2graph(rawgraph):
    """Converts raw graph to graph."""
    graph = {}
    for parent, children in rawgraph.items():
        graph[parent[0]] = []
        for child in children:
            graph[parent[0]].append(child[0])

    return graph

def graph2tokenized_sent(graph, tokenized_sent):
    """Converts graph to tokenized sentence of 4-tuples.

    Args:
        graph: A graph dictionary.
        tokenized_sent: A list of 3-tuples, e.g.:
            `[(0, '<ROOT>', '<ROOT-POS>'),
              (1, 'Not', 'RB'),
              (2, 'this', 'DT'),
              (3, 'year', 'NN'),
              (4, '.', '.')]`
    Returns:
        A list of 4-tuples sentence.
    """
    result = [t[:3] for t in tokenized_sent]
    result[0] = result[0] + (-1, )
    for parent_idx, children in graph.items():
        for child_idx in children:
            result[child_idx] = result[child_idx] + (parent_idx, )
    return result

def print_log(dict_):
    """Helper for log during training."""
    buff = "\r" + "|".join([str(k) + ':' + str(v) for k, v in dict_.items()])
    sys.stdout.write(buff)
    sys.stdout.flush()

class DependencyParser:
    """A class that parses input, trains model and performs predictions.
    
    ## These are a few notes on notatiosn:
    * `tokenized_sent` refers to a list of 3-tuples or 4-tuples. For
    example, the following refers to `tokenized_tuple` as train data.
                    `[(0, '<ROOT>', '<ROOT-POS>', -1),
                      (1, 'Not', 'RB', 3),
                      (2, 'this', 'DT', 3),
                      (3, 'year', 'NN', 0),
                      (4, '.', '.', 3)]`
    Removing the last element in each tuple (token head) is refered
    to as `tokenized_tuple` for inference.
    
    * `rawgraph` refers to a graph having a following form (continuation
        of the previous example):
        `{(0, '<ROOT>', '<ROOT-POS>'): [(3, 'year', 'NN')],
          (1, 'Not', 'RB'): [],
          (2, 'this', 'DT'): [],
          (3, 'year', 'NN'): [(1, 'Not', 'RB'), (2, 'this', 'DT'), (4, '.', '.')],
          (4, '.', '.'): []}`
    
    * `graph` refers to a following object
        `{0: [3], 1: [], 2: [], 3: [1, 2, 4], 4: []}`
    
    * `featuregraph` is a graph that stores edges as keys and indicies
        as values. Indices represent an indicator vector. By taking
        the previous example, we get:
        `{(0, 3): [0, 1, 2, 4362, 4363, 4364, 716, 717, 76],
          (3, 1): [711, 712, 90, 4365, 4366, 3243, 4367, 4368, 300],
          (3, 2): [711, 712, 90, 264, 713, 93, 266, 267, 96],
          (3, 4): [711, 712, 90, 4369, 4370, 4371, 65, 66, 67]}`

    """
    def __init__(self, raw_data, edge2rawfeature=None, min_feature_count=1):
        """Instantiates a new `DependencyParser` instance.

        Args:
            raw_data: A sentences splitable by `\n\n` character.
                For example, the sentence must have following form:
                `'1\t@\t_\tIN\t_\t_\t0\tROOT\t_\t_'`
            edge2rawfeature: A callable taking two argument tokens
                `parent` and `child` and returning a list of
                tuples representing feature names and values. For
                example, see `edge2rawfeature1()` method.
            min_feature_count: A integer specifying a minimum
                times a feature should occure to be included as
                feature. It excludes rare features.
        """

        self.edge2rawfeature = (DependencyParser.edge2rawfeature1
                                if edge2rawfeature is None
                                else edge2rawfeature)

        tokenized_sents = raw_data2tokenized_sents(raw_data)
        feature_counts = {}
        
        for tokenized_sent in tokenized_sents:
            graph = tokenized_sent2rawgraph(tokenized_sent)
            features = []
            for parent, children in graph.items():
                for child in children:
                    features += self.edge2rawfeature(parent, child)
            for f in features:
                if f in feature_counts:
                    feature_counts[f] += 1
                else:
                    feature_counts[f] = 1

        features = [f for f, v in feature_counts.items() if v >= min_feature_count]

        self.n_features = len(features)
        self._mapping = {}
        self._inverse_mapping = {}
        for f in features:
            self._mapping[f] = len(self._mapping)
            self._inverse_mapping[len(self._mapping) - 1] = f

        self.w = np.zeros(shape=self.n_features)

    @staticmethod
    def edge2rawfeature1(parent_token, child_token):
        features = []
        features.append(('p-word, p-pos', parent_token[1], parent_token[2]))
        features.append(('p-word', parent_token[1]))
        features.append(('p-pos', parent_token[2]))
        features.append(('p-pos, c-word, c-pos', parent_token[2], child_token[1], child_token[2]))
        features.append(('p-word, p-pos, c-pos', parent_token[1], parent_token[2], child_token[2]))
        features.append(('p-pos, c-pos', parent_token[2], child_token[2]))
        features.append(('c-word, c-pos', child_token[1], child_token[2]))
        features.append(('c-word', child_token[1]))
        features.append(('c-pos', child_token[2]))
        return features

    @staticmethod
    def edge2rawfeature2(parent_token, child_token):
        features = DependencyParser.edge2rawfeature1(parent_token, child_token)
        features.append(('distance', abs(parent_token[0] - child_token[0])))
        features.append(('p-word, p-pos, c-word, c-pos',
                         parent_token[1],
                         parent_token[2],
                         child_token[1],
                         child_token[2]))
        features.append(('p-word, c-word, c-pos',
                         parent_token[1],
                         child_token[1],
                         child_token[2]))
        features.append(('p-word, p-pos, c-word',
                         parent_token[1],
                         parent_token[2],
                         child_token[1]))
        features.append(('p-word, c-word', parent_token[1], child_token[1]))
        features.append(('parent_before_child', parent_token[0] < child_token[0]))
        
        if len(parent_token[1]) >= 4 and len(child_token[1]) >= 4:
            features.append(('4p-word, 4c-word', parent_token[1][:4], child_token[1][:4]))
            features.append(('p-word4, c-word4', parent_token[1][-4:], child_token[1][-4:]))

        if len(parent_token[1]) >= 3 and len(child_token[1]) >= 3:
            features.append(('3p-word, 3c-word', parent_token[1][:3], child_token[1][:3]))
            features.append(('p-word3, c-word3', parent_token[1][-3:], child_token[1][-3:]))

        if len(parent_token[1]) >= 2 and len(child_token[1]) >= 2:
            features.append(('2p-word, 2c-word', parent_token[1][:2], child_token[1][:2]))
            features.append(('p-word2, c-word3', parent_token[1][-2:], child_token[1][-2:]))
        


        return features

    def rawgraph2featuregraph(self, rawgraph):
        """Converts rawgraph to featuregraph."""
        featuregraph = {}
        for parent, children in rawgraph.items():
            for child in children:
                if (parent, child) not in featuregraph:
                    featuregraph[(parent[0], child[0])] = []
                features = self.edge2rawfeature(parent, child)
                for f in features:
                    if f in self._mapping:
                        featuregraph[(parent[0], child[0])].append(self._mapping[f])
        return featuregraph

    def prepare_train_data(self, tokenized_sents):
        """Converts `tokenized_sent` to an object for simplified training."""
        data = []

        for tokenized_sent in tokenized_sents:
            rawgraph = tokenized_sent2rawgraph(tokenized_sent)
            graph = rawgraph2graph(rawgraph)
            template_rawgraph = tokenized_sent2rawclique(tokenized_sent)
            template_graph = rawgraph2graph(template_rawgraph)
            feature_template_graph = self.rawgraph2featuregraph(template_rawgraph)
            feature_actual_graph = self.rawgraph2featuregraph(rawgraph)
            data.append({
                    'actual_rawgraph': rawgraph,
                    'actual_graph':graph,
                    'feature_actual_graph':feature_actual_graph,
                    'template_rawgraph': template_rawgraph,
                    'template_graph': template_graph,
                    'feature_template_graph': feature_template_graph,
                    'tokenized_sent': tokenized_sent
                })
        return data

    def predict(self, tokenized_sent):
        """Predicts dependencies of a tokenized sentence.

        Args:
            tokenized_sent: A list of 3-tuples e.g.
                `[(0, 'ROOT', 'ROOT_POS'): [(1, 'Hello', 'NN'), ...]`

        Returns:
            A list of 4-tuples e.g.
                `[(0, 'ROOT', 'ROOT_POS', -1): [(1, 'Hello', 'NN', 2), ...]`
        """
        template_rawgraph = tokenized_sent2rawclique(tokenized_sent)
        template_graph = rawgraph2graph(template_rawgraph)
        feature_template_graph = self.rawgraph2featuregraph(template_rawgraph)

        score = GetScore()
        score.w = self.w
        score.feature_graph = feature_template_graph
        digraph = Digraph(template_graph, score.get_score)
        mst = digraph.mst()
        predicted_graph = mst.successors
        predicted_tokenized_sent = graph2tokenized_sent(predicted_graph,
                                                        tokenized_sent)
        return predicted_tokenized_sent

    def optimize(self, train_data, test_data=None, n_epochs=100, log_interval=200, lr=1, fname_prefix='log_'):
        """Structured perceptron optimization.

        Args:
            data: A list of dictionaries created by `prepare_train_data()`
                function.
            n_epochs: Number of epochs over all data.
            log_interval: An interval after each the current accuracy
                updated on the screen.
            lr: Learning rate.
            fname_prefix: A prefix that is appended to the stored
                train log and weights pickle files.
        """
        global_step = 0
        train_log = [0.0]
        accuracy = []
        test_accuracy = [0]
        score = GetScore()
        score.w = self.w

        for epoch in range(n_epochs):
            random.shuffle(train_data)
            for d in train_data:
                global_step += 1
                score.feature_graph = d['feature_template_graph']
                digraph = Digraph(d['template_graph'],
                                  score.get_score,)
                mst = digraph.mst()
                predicted_graph = mst.successors

                predicted_tokenized_sent = graph2tokenized_sent(predicted_graph,
                                                                     d['tokenized_sent'])
                predicted_rawgraph = tokenized_sent2rawgraph(predicted_tokenized_sent)
                feature_predicted_graph = self.rawgraph2featuregraph(predicted_rawgraph)
                results = [p == a for p, a in zip(predicted_tokenized_sent[1:],
                                                  d['tokenized_sent'][1:])]

                accuracy.append(sum(results)/len(results))

                if accuracy[-1] != 1:
                    for _, indices in feature_predicted_graph.items():
                        self.w[indices] -= lr
                    for _, indices in d['feature_actual_graph'].items():
                        self.w[indices] += lr

                if global_step % log_interval == 0 or global_step in [1, 5, 10]:
                    average_accuracy = np.mean(accuracy)
                    print_log({'acc': average_accuracy,
                               'acc_test': test_accuracy[-1],
                               'epoch': epoch,
                               'step': global_step})
                    train_log.append(average_accuracy)
                    accuracy = []
                
                del digraph
                del mst
                del predicted_graph
                del predicted_tokenized_sent
                del predicted_rawgraph
                del feature_predicted_graph
            if accuracy:
                train_log.append(np.mean(accuracy))
                accuracy = []
            if test_data is not None:
                test_accuracy.append(self.compute_accuracy(test_data))
            gc.collect()

        with open(fname_prefix + 'weights.pkl', 'wb') as fo:
            pickle.dump(self.w, fo, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fname_prefix + 'accuracy.pkl', 'wb') as fo:
            pickle.dump(train_log, fo, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fname_prefix + 'test_accuracy.pkl', 'wb') as fo:
            pickle.dump(test_accuracy, fo, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_accuracy(self, tokenized_sents):
        accuracy = []
        for actual_sent in tokenized_sents:
            predicted_sent = self.predict(actual_sent)
            accuracy.append(compute_accuracy(actual_sent[1:],
                                             predicted_sent[1:]))
        return np.mean(accuracy)


class GetScore:
    """Helper for calculating score of the edge."""
    def get_score(self, x, y):
        edge = (x, y)
        if edge in self.feature_graph:
            return self.w[self.feature_graph[edge]].sum()
        else:
            return 0.0

def edge2rawfeature10(parent_token, child_token):
    """Creates features for the second model."""
    features = []
    features.append(('p-word, p-pos', parent_token[1], parent_token[2]))
    features.append(('p-word', parent_token[1]))
    features.append(('p-pos', parent_token[2]))
    features.append(('p-pos, c-word, c-pos',
                     parent_token[2],
                     child_token[1],
                     child_token[2]))
    features.append(('p-word, p-pos, c-pos',
                     parent_token[1],
                     parent_token[2],
                     child_token[2]))
    features.append(('p-pos, c-pos',
                     parent_token[2],
                     child_token[2]))
    features.append(('c-word, c-pos',
                     child_token[1],
                     child_token[2]))
    features.append(('c-word', child_token[1]))
    features.append(('c-pos', child_token[2]))
    
    features.append(('p-word, p-pos, c-word, c-pos',
                     parent_token[1],
                     parent_token[2],
                     child_token[1],
                     child_token[2]))
    features.append(('p-word, c-word, c-pos',
                     parent_token[1],
                     child_token[1],
                     child_token[2]))
    features.append(('p-word, p-pos, c-word',
                     parent_token[1],
                     parent_token[2],
                     child_token[1]))
    features.append(('p-word, c-word', parent_token[1], child_token[1]))
    features.append(('parent_before_child', parent_token[0] < child_token[0]))

    features.append(('p-pos, c-pos, dist',
                     parent_token[2],
                     child_token[2],
                     abs(parent_token[0] - child_token[0])))
    ###################################################################
    features.append(('p-word, c-word, dist',
                     parent_token[1],
                     child_token[1],
                     abs(parent_token[0] - child_token[0])))
    ###################################################################
    if len(parent_token[1]) >= 4 and len(child_token[1]) >= 4:
        features.append(('4p-word, 4c-word', parent_token[1][:4], child_token[1][:4]))
        features.append(('p-word4, c-word4', parent_token[1][-4:], child_token[1][-4:]))

    if len(parent_token[1]) >= 3 and len(child_token[1]) >= 3:
        features.append(('3p-word, 3c-word', parent_token[1][:3], child_token[1][:3]))
        features.append(('p-word3, c-word3', parent_token[1][-3:], child_token[1][-3:]))

    if len(parent_token[1]) >= 2 and len(child_token[1]) >= 2:
        features.append(('2p-word, 2c-word', parent_token[1][:2], child_token[1][:2]))
        features.append(('p-word2, c-word3', parent_token[1][-2:], child_token[1][-2:]))
    
    ###################################################################
    if len(parent_token[1]) >= 4 and len(child_token[1]) >= 4:
        features.append(('4p-word, 4c-word, p-pos', parent_token[1][:4], child_token[1][:4], parent_token[2]))
        features.append(('p-word4, c-word4, p-pos', parent_token[1][-4:], child_token[1][-4:], parent_token[2]))

    if len(parent_token[1]) >= 3 and len(child_token[1]) >= 3:
        features.append(('3p-word, 3c-word, p-pos', parent_token[1][:3], child_token[1][:3], parent_token[2]))
        features.append(('p-word3, c-word3, p-pos', parent_token[1][-3:], child_token[1][-3:], parent_token[2]))

    if len(parent_token[1]) >= 2 and len(child_token[1]) >= 2:
        features.append(('2p-word, 2c-word, p-pos', parent_token[1][:2], child_token[1][:2], parent_token[2]))
        features.append(('p-word2, c-word3, p-pos', parent_token[1][-2:], child_token[1][-2:], parent_token[2]))
    
    ###################################################################
    if len(parent_token[1]) >= 4 and len(child_token[1]) >= 4:
        features.append(('4p-word, 4c-word, c-pos', parent_token[1][:4], child_token[1][:4], child_token[2]))
        features.append(('p-word4, c-word4, c-pos', parent_token[1][-4:], child_token[1][-4:], child_token[2]))

    if len(parent_token[1]) >= 3 and len(child_token[1]) >= 3:
        features.append(('3p-word, 3c-word, c-pos', parent_token[1][:3], child_token[1][:3], child_token[2]))
        features.append(('p-word3, c-word3, c-pos', parent_token[1][-3:], child_token[1][-3:], child_token[2]))

    if len(parent_token[1]) >= 2 and len(child_token[1]) >= 2:
        features.append(('2p-word, 2c-word, c-pos', parent_token[1][:2], child_token[1][:2], child_token[2]))
        features.append(('p-word2, c-word3, c-pos', parent_token[1][-2:], child_token[1][-2:], child_token[2]))
    
    
    return features