"""
A debugging script that uses a python feed dictionary to serve data to the tensorflow graph. This is really inefficient,
but allows the gradients and tensors to be exported after each iteration
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
import datetime
import run_detectors
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import visualisation

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=1.8)]


# construct input-output pairs
class Params:
    def __init__(self, batch_size, embedding_size, neg_samples, skip_window, num_pairs, logging_interval,
                 initial_learning_rate):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.neg_samples = neg_samples
        self.skip_window = skip_window
        self.num_pairs = num_pairs  # the total number of (input, output) pairs to train with
        self.logging_interval = logging_interval  # the number of batches between loss logs
        self.initial_learning_rate = initial_learning_rate
        assert self.num_pairs / self.batch_size == int(self.num_pairs / self.batch_size)


class Graph2Vecs():
    def __init__(self, unigrams, params):
        # Set the parameters
        self.params = params
        self.vocab_size = len(unigrams)
        self.batch_size = params.batch_size
        self.embedding_size = params.embedding_size  # Dimension of the embedding vector.
        self.num_samples = self.batch_size * params.neg_samples
        self.epochs_to_train = 1
        self.initial_learning_rate = params.initial_learning_rate
        self.global_step = tf.Variable(0, name="global_step")
        self.n_pairs = 0  # progress counter
        # self.words_to_train = float(words_per_epoch * self.epochs_to_train)
        self.examples = tf.placeholder(tf.int32, shape=[self.batch_size], name='examples')
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='labels')
        self.lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.unigrams = unigrams
        # Add opp to save variables
        self.saver = tf.train.Saver()

        true_logits, sampled_logits = self.forward(self.examples, self.labels)
        self.loss = self.nce_loss(true_logits, sampled_logits)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train = optimizer.minimize(self.loss, global_step=self.global_step,
                                        gate_gradients=optimizer.GATE_NONE)

    # update the learning rate
    def update_lr(self, n_pairs):
        """
        linear decay of learning rate. Update the optimizer after each batch
        :param n_words: the number of output-context pairs seen so far
        :return:
        """
        lr = self.initial_learning_rate * max(
            0.0001, 1.0 - float(n_pairs) / self.params.num_pairs)
        return lr

    # Define the computational graph
    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / self.embedding_size
        self.emb = tf.Variable(
            tf.random_uniform(
                [self.vocab_size, self.embedding_size], -init_width, init_width),
            name="emb")

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([self.vocab_size, self.embedding_size]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [self.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,  # set to True if all the samples need to be unique
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=self.unigrams.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(self.emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    # define the loss function
    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size
        return nce_loss_tensor

        # # put it all together
        # def build_graph(self, examples, labels):
        #     """
        #     build the forward graph, the loss function and the optimizer
        #     :param examples: training inputs
        #     :param labels: training labels
        #     :return:
        #     """
        #     true_logits, sampled_logits = self.forward(examples, labels)
        #     loss = self.nce_loss(true_logits, sampled_logits)
        #     # tf.contrib.deprecated.scalar_summary("NCE loss", loss)
        #     self.loss = loss
        #     self.train = self.optimize(loss)

def initialize_embedding():
    pass

# produce batch of data

def generate_batch(skip_window, data, batch_size):
    """
    A generator that produces the next batch of examples and labels
    :param skip_window: The largest distance between an example and a label
    :param data:  the random walks
    :param batch_size: the number of (input, output) pairs to return
    :return:
    """
    row_index = 0
    examples = []
    labels = []
    while True:
        sentence = data[row_index, :]
        for pos, word in enumerate(sentence):
            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - skip_window)
            # enumerate takes a second arg, which sets the starting point, this makes pos and pos2 line up
            for pos2, word2 in enumerate(sentence[start: pos + skip_window + 1], start):
                if pos2 != pos:
                    examples.append(word)
                    labels.append([word2])
                    if len(examples) == batch_size:
                        yield examples, labels
                        examples = []
                        labels = []
        row_index = (row_index + 1) % data.shape[0]


def edge_list2pairs(edge_list_path, epochs):
    """
    converts an edgelist into a a tuple of Skipgram inputs of only first order connections
    :param edge_list_path: the path to the edgelist text file, which has two columns [out_idx, in_idx]
    :param epochs: The number of pairs per input edge to return
    :return: A tuple of (list(examples), list(list(labels))) as required by the tf word2vec library
    """
    df1 = pd.read_csv(edge_list_path, sep=' ', header=None)
    # edges are undirected so need to flip them
    df2 = pd.DataFrame({0: df1[1], 1: df1[0]})
    edge_list = pd.concat((df1, df2), axis=0).values
    all_pairs = np.tile(edge_list, (epochs, 1))
    return all_pairs


def generate_edge_batch(data, batch_size):
    """
    A generator that produces the next batch of examples and labels
    :param skip_window: The largest distance between an example and a label
    :param data:  the edgelist in a numpy array of shape (None, 2)
    :param batch_size: the number of (input, output) pairs to return
    :return:
    """
    row_index = 0
    examples = []
    labels = []
    while True:
        row = data[row_index, :]
        examples.append(row[0])
        labels.append([row[1]])
        if len(examples) == batch_size:
            yield examples, labels
            examples = []
            labels = []
        row_index = (row_index + 1) % data.shape[0]


def main(outpath, walks, unigrams, params):
    # initialise the graph
    graph = tf.Graph()
    # run the tensorflow session
    with tf.Session(graph=graph) as session:
        # Define the training data
        model = Graph2Vecs(unigrams, params)

        # initialize all variables in parallel
        tf.global_variables_initializer().run()
        _ = [print(v) for v in tf.global_variables()]

        s = datetime.datetime.now()
        print("Initialized")
        # define batch generator
        batch_gen = generate_batch(params.skip_window, walks, params.batch_size)
        data = edge_list2pairs('local_resources/zachary_karate/karate.edgelist', 100)
        edge_gen = generate_edge_batch(data, params.batch_size)
        average_loss = 0
        n_pairs = 0
        num_steps = params.num_pairs / params.batch_size
        print('running for ', num_steps, ' steps')
        for step in xrange(int(num_steps)):
            s_batch = datetime.datetime.now()
            if step < num_steps / 2:
                batch_inputs, batch_labels = edge_gen.next()
            else:
                batch_inputs, batch_labels = batch_gen.next()
            lr = model.update_lr(n_pairs)
            feed_dict = {model.lr: lr, model.examples: batch_inputs, model.labels: batch_labels}
            _, loss_val = session.run([model.train, model.loss], feed_dict=feed_dict)
            average_loss += loss_val
            n_pairs += params.batch_size
            if step % params.logging_interval == 0:
                if step > 0:
                    average_loss /= params.logging_interval
                # The average loss is an estimate of the loss over the last 2000 batches.
                runtime = datetime.datetime.now() - s_batch
                print("Average loss at step ", step, ": ", average_loss, 'learning rate is', lr, 'ran in', runtime)
                s_batch = datetime.datetime.now()
                average_loss = 0
        # final_embeddings = normalized_embeddings.eval()
        final_embedding = model.emb.eval()
        np.savetxt(outpath, final_embedding)
        # saver.save(session, 'tf_out/test.ckpt')
        # ckpt = tf.train.get_checkpoint_state('tf_out')
        # saver.restore(session, ckpt.model_checkpoint_path)
        # np.savetxt('resources/test/tf_test2.csv', emb.eval())
        print('ran in {0} s'.format(datetime.datetime.now() - s))
        return final_embedding


def get_vocab_size(adj_path, bipartite):
    """
    Get the number of vertices in the graph (equivalent to the NLP vocab size)
    :param adj_path: the path to the sparse CSR adjacency matrix
    :param bipartite: True if the graph is bipartite
    :return: an integer vocab_size
    """
    adj = utils.read_pickle(adj_path)
    vocab_size = adj.shape[0]
    if bipartite:
        vocab_size += adj.shape[1]
    return vocab_size


def twitter_age_scenario():
    # Read the data
    walks = pd.read_csv('resources/test/node2vec/walks_1.0_1.0.csv', header=None).values
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    vocab_size = get_vocab_size(x_path, bipartite=True)
    # define the noise distribution
    _, unigrams = np.unique(walks, return_counts=True)
    params = Params()
    main('resources/test/tf.emd', walks, unigrams, vocab_size, params)


def karate_results(embeddings, names, n_reps, train_size):
    deepwalk_path = 'local_resources/zachary_karate/size8_walks1_len10.emd'

    y_path = 'local_resources/zachary_karate/y.p'
    x_path = 'local_resources/zachary_karate/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    # names = [['embedding'], ['logistic']]

    names.append(['logistics'])

    # x_deepwalk = utils.read_embedding(deepwalk_path, target)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    # X = [normalize(embedding, axis=0), normalize(x, axis=0)]
    X = embeddings + [normalize(x, axis=0)]
    # names = ['embedding']
    # X = embedding

    results = []
    for exp in zip(X, names):
        tmp = run_detectors.run_experiments(exp[0], y, exp[1], classifiers, n_reps, train_size)
        results.append(tmp)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/karate/tf_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/karate/tf_micro_pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = 'results/karate/tf_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/karate/tf_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)
    return results


def karate_scenario():
    walks = pd.read_csv('local_resources/zachary_karate/walks1_len10_p1_q1.csv', header=None).values
    x_path = 'local_resources/zachary_karate/X.p'
    y_path = 'local_resources/zachary_karate/y.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    vocab_size = get_vocab_size(x_path, bipartite=False)
    print('vocab of size: ', vocab_size)
    # define the noise distribution
    elems, unigrams = np.unique(walks, return_counts=True)
    print('unigram distribution', zip(elems, unigrams))
    embeddings = []
    names = []
    n_reps = 10
    train_size = 4  # the number of labelled points to use
    size = 2
    for window in xrange(1, 10):
        params = Params(batch_size=4, embedding_size=size, neg_samples=5, skip_window=3, num_pairs=1500,
                        logging_interval=100,
                        initial_learning_rate=0.2)
        embedding = main('local_resources/zachary_karate/tf.emd', walks, unigrams, params)
        embeddings.append(embedding)
        names.append(['window' + str(window)])
        if size == 2:
            visualisation.plot_embedding(embedding, y, 'results/karate/figs/window_' + str(
                window) + '_' + utils.get_timestamp() + '.pdf')
    karate_results(embeddings, names, n_reps, train_size)


def generate_embeddings(name, emd_reps, det_reps, params, walks, unigrams, train_size):
    """
    Create a load of embeddings that are identical except for the random initialisations
    :param name:
    :param emd_reps:
    :param det_reps:
    :param params:
    :param walks:
    :param unigrams:
    :param train_size:
    :return:
    """
    embeddings = []
    names = []
    for window in xrange(0, emd_reps):
        embedding = main('local_resources/zachary_karate/tf.emd', walks, unigrams, params)
        embeddings.append(embedding)
        names.append([name + str(window)])
    results = run_embedding_array(embeddings, names, det_reps, train_size)
    return results


def run_embedding_array(embeddings, names, n_reps, train_size):
    """
    As embeddings show significant variation we must compare many embeddings with the same params to ascertain quality
    :param embeddings:
    :param names:
    :param n_reps:
    :param train_size:
    :return: A tuple of pandas DataFrames (macro, micro)
    """
    y_path = 'local_resources/zachary_karate/y.p'
    x_path = 'local_resources/zachary_karate/X.p'

    x, y = utils.read_data(x_path, y_path, threshold=0)

    results = []
    for exp in zip(embeddings, names):
        tmp = run_detectors.run_experiments(exp[0], y, exp[1], classifiers, n_reps, train_size)
        results.append(tmp)

    all_results = utils.merge_results(results)

    return all_results


def compare_embeddings():
    emd_reps = 10  # number of times to generate the embeddings
    det_reps = 10  # number of times to repeat the classification
    train_size = 4  # number of training examples
    size = 2  # the number of dimensions to embed
    walks = pd.read_csv('local_resources/zachary_karate/walks1_len10_p1_q1.csv', header=None).values
    p1 = Params(batch_size=4, embedding_size=size, neg_samples=5, skip_window=3, num_pairs=1500,
                logging_interval=100,
                initial_learning_rate=0.2)
    p2 = Params(batch_size=4, embedding_size=size, neg_samples=8, skip_window=3, num_pairs=1500,
                logging_interval=100,
                initial_learning_rate=0.2)
    param_arr = [p1, p2]
    elems, unigrams = np.unique(walks, return_counts=True)
    names = ['neg5', 'neg8']
    results = []
    for name, params in zip(names, param_arr):
        result = generate_embeddings(name, emd_reps, det_reps, params, walks, unigrams, train_size)
        results.append(result)

    means, tests = utils.array_stats_test(results)
    tests[0].to_csv('results/karate/tf_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/karate/tf_micro_pvalues' + utils.get_timestamp() + '.csv')
    print('results', means)
    means_path = 'results/karate/tf_means' + utils.get_timestamp() + '.csv'
    means.to_csv(means_path, index=True)
    all_results = utils.merge_results(results)
    macro_path = 'results/karate/tf_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/karate/tf_micro' + utils.get_timestamp() + '.csv'
    all_results[0].to_csv(macro_path, index=True)
    all_results[1].to_csv(micro_path, index=True)


if __name__ == '__main__':
    s = datetime.datetime.now()
    karate_scenario()
    print(datetime.datetime.now() - s)
