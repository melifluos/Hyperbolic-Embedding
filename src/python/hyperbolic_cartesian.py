"""
neural embeddings using 2d polar co-ordinates
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd
import utils
import numpy as np
import datetime
import run_detectors
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from preprocessing import Customers
from sklearn.preprocessing import normalize
import threading
import time
import run_detectors
import os

# add custom operator
generate_batch = tf.load_op_library('../cpp/generate_batch.so')

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=1.8)]


# construct input-output pairs
class Params:
    def __init__(self, filepath, batch_size, embedding_size, neg_samples, skip_window, num_pairs, statistics_interval,
                 initial_learning_rate, save_path='', epochs=1, concurrent_steps=10):
        self.filepath = filepath
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_samples = neg_samples  # The number of negative samples for the entire batch - this should scale with batch size
        self.skip_window = skip_window
        # num_pairs NOT CURRENTLY IN USE
        self.num_pairs = num_pairs  # the total number of (input, output) pairs to train with.
        self.statistics_interval = statistics_interval  # the number of seconds between snapping
        self.learning_rate = initial_learning_rate
        self.epochs_to_train = epochs
        self.concurrent_steps = concurrent_steps
        self.save_path = save_path
        # assert self.num_pairs / self.batch_size == int(self.num_pairs / self.batch_size)


class cust2vec():
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.initialisation = None
        # self.examples
        # self.labels

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-
        #  is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def sinh(self, x):
        return 0.5 * (tf.subtract(tf.exp(x), tf.exp(-x)))

    def modify_grads(self, grads, emb):
        """
        The tensor flow autograd gives us Euclidean gradients. Here we multiply by (1/4)(1-||emb||^2)^2
        to convert to the hyperbolic gradient
        :param grads: a list of tuples of [(grads, name),...]
        :param emb: A tensor embedding
        :return: The hyperbolic gradient
        """
        scaled_grads = []
        for grad, name in grads:
            vecs = tf.nn.embedding_lookup(emb, grad.indices)
            norm_squared = tf.square(tf.norm(vecs, axis=0))
            hyperbolic_factor = 0.25 * tf.square(1 - norm_squared)
            g = tf.multiply(grad.values, hyperbolic_factor)
            # g_clip = tf.clip_by_value(g, -0.1, 0.1)
            scaled_grad = tf.IndexedSlices(g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        # scaled_theta_grad = [(tf.clip_by_value(tf.scatter_div(g, g.indices, radius), -1, 1), v) for g, v in grads]
        return scaled_grads

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        opts = self._options
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)
        sm_b_grad = optimizer.compute_gradients(loss, [self.sm_b])
        emb_grad = optimizer.compute_gradients(loss, [self.emb])
        sm_w_t_grad = optimizer.compute_gradients(loss, [self.sm_w_t])

        self.emb_grad = emb_grad
        self.sm_w_t_grad = sm_w_t_grad

        modified_emb_grad = self.modify_grads(emb_grad, self.emb)
        modified_sm_w_t_grad = self.modify_grads(sm_w_t_grad, self.sm_w_t)
        # theta_out_clipped = tf.clip_by_value(modified_theta_out, -1, 1, name="theta_out_clipped")
        self.modified_emb_grad = modified_emb_grad
        self.modified_sm_w_t_grad = modified_sm_w_t_grad
        gv = sm_b_grad + modified_emb_grad + modified_sm_w_t_grad
        self._train = optimizer.apply_gradients(gv, global_step=self.global_step)

    def build_graph(self):
        """
        construct the graph's forward pass, loss, optimizer and training procedure
        :return:
        """
        opts = self._options
        print('building graph')
        (words, counts, words_per_epoch, self._epoch, self._words, self.examples,
         self.labels) = generate_batch.skipgram_word2vec(filename=opts.filepath, batch_size=opts.batch_size,
                                                         window_size=opts.skip_window)

        # set the number of words in each epoch
        (opts.vocab_words, opts.vocab_counts, opts.words_per_epoch) = self._session.run(
            [words, counts, words_per_epoch])
        lookup = pd.DataFrame(index=opts.vocab_words, data=opts.vocab_counts, columns=['counts'])
        lookup.to_csv('../../local_results/tf_index.csv')
        # print ('examples: ', examples, '\n', 'labels: ', labels)
        opts.vocab_size = len(opts.vocab_words)
        print(opts.words_per_epoch, ' words per epoch')
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i
        true_logits, sampled_logits = self.forward(self.examples, self.labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar("NCE loss", loss)
        self._loss = loss
        self.optimize(loss)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        # Add opp to save variables
        self.saver = tf.train.Saver()

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        print('thread sees initial epoch: ', initial_epoch)
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            # print('thread sees epoch: ', epoch)
            if epoch != initial_epoch:
                break

    def train(self):
        opts = self._options
        initial_epoch, initial_words = self._session.run([self._epoch, self._words])
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
        workers = []
        for thread_counter in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            # print('starting thread: ', thread_counter)
            t.start()
            workers.append(t)
        print('running epoch: ', initial_epoch, ' on ', opts.concurrent_steps, ' threads')
        last_words, last_time, last_summary_time = initial_words, time.time(), 0
        while True:
            time.sleep(opts.statistics_interval)  # Reports our progress once a while.
            (epoch, step, loss, words, lr, examples, labels, grads, mod_grads) = self._session.run(
                [self._epoch, self.global_step, self._loss, self._words, self._lr, self.examples,
                 self.labels, self.emb_grad, self.sm_w_t_grad])
            assert len(examples) == opts.batch_size
            assert len(labels) == opts.batch_size
            # print('global step: ', step)
            # if step % 1000 == 0:
            now = time.time()
            rate = float(words - last_words) / float(now - last_time)
            last_words = words
            last_time = now
            print("Epoch {0} Step {1}: lr = {2} loss = {3} words/sec = {4} words processed = {5}\n".format(epoch,
                                                                                                           step, lr,
                                                                                                           loss,
                                                                                                           rate,
                                                                                                           words))
            # print("labels: ", labels, " examples: ", examples)
            # print("grads are {0}".format(grads))
            #
            # print("modified grads are {0}".format(mod_grads))
            summary_str = self._session.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            self.saver.save(self._session, os.path.join(opts.save_path, "model.ckpt"),
                            global_step=step.astype(int))
            if epoch != initial_epoch:
                break
        for t in workers:
            t.join()

    def poincare_initialisation(self):
        """
        initialise points uniformly within the poincare disc model
        :return:
        """
        opts = self._options
        r = np.sqrt(np.random.uniform(low=0, high=1, size=opts.vocab_size))  # radius
        theta = np.random.uniform(low=0, high=2 * np.pi, size=opts.vocab_size)  # angle
        arr = np.zeros(shape=(opts.vocab_size, 2), dtype=np.float32)
        arr[:, 0] = r
        arr[:, 1] = theta
        return arr

    def arctanh(self, x):
        """
        Get the inverse hyperbolic tangent
        :param x: A tensor
        :return:
        """
        return 0.5 * tf.log(tf.divide(1 + x, 1 - x))

    def inner_prod(self, r_in, r_out, theta_in, theta_out):
        """
        Takes the hyperbolic inner product
        :param r_in: radius in the input embedding
        :param r_out: radius in the output embedding
        :param theta_in:
        :param theta_out:
        :return:
        """
        cosine = tf.cos(theta_in - theta_out)
        radius = tf.multiply(r_in, r_out)
        return tf.multiply(cosine, radius)

    def tensor_inner_prod(self, r_example, r_sample, theta_example, theta_sample):
        """
        Calculate the inner product between the examples and the negative samples
        :param r_example:
        :param r_sample:
        :param theta_example:
        :param theta_sample:
        :return:
        """
        radius_term = tf.multiply(r_example[:, None], r_sample[None, :])
        cos_term = tf.cos(theta_example[:, None] - theta_sample[None, :])
        return tf.squeeze(tf.multiply(cos_term, radius_term))

    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        # Embedding: [vocab_size, emb_dim]
        opts = self._options
        with tf.name_scope('model'):
            init_width = 0.5 / opts.embedding_size
            emb = np.random.uniform(low=-init_width, high=init_width,
                                    size=(opts.vocab_size, opts.embedding_size)).astype(np.float32)

            self.emb = tf.Variable(
                tf.random_uniform(
                    [opts.vocab_size, opts.embedding_size], -init_width, init_width),
                name="emb")

            emb_hist = tf.summary.histogram('embedding', emb)

            # Softmax weight: [vocab_size, emb_dim]. Transposed.
            self.sm_w_t = tf.Variable(
                tf.zeros([opts.vocab_size, opts.embedding_size]),
                name="sm_w_t")

            # smw_hist = tf.summary.histogram('softmax weight', self.sm_w_t)

            # Softmax bias: [emb_dim].
            self.sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
            smb_hist = tf.summary.histogram('softmax bias', self.sm_b)

            # Create a variable to keep track of the number of batches that have been fed to the graph
            self.global_step = tf.Variable(0, name="global_step")

        with tf.name_scope('input'):
            # Nodes to compute the nce loss w/ candidate sampling.
            labels_matrix = tf.reshape(
                tf.cast(labels,
                        dtype=tf.int64),
                [opts.batch_size, 1])

            # Embeddings for examples: [batch_size, emb_dim]
            example_emb = tf.nn.embedding_lookup(self.emb, examples)
            example_hist = tf.summary.histogram('input embeddings', example_emb)

            # Weights for labels: [batch_size, emb_dim]
            true_w = tf.nn.embedding_lookup(self.sm_w_t, labels)
            # Biases for labels: [batch_size, 1]
            true_b = tf.nn.embedding_lookup(self.sm_b, labels)

        with tf.name_scope('negative_samples'):
            # Negative sampling.
            sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=opts.num_samples,
                unique=True,
                range_max=opts.vocab_size,
                distortion=0.75,
                unigrams=opts.vocab_counts.tolist()))

            # Weights for sampled ids: [num_sampled, emb_dim]
            sampled_w = tf.nn.embedding_lookup(self.sm_w_t, sampled_ids)
            # Biases for sampled ids: [num_sampled, 1]
            sampled_b = tf.nn.embedding_lookup(self.sm_b, sampled_ids)

            # True logits: [batch_size, 1]
            true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

            # Sampled logits: [batch_size, num_sampled]
            # We replicate sampled noise labels for all examples in the batch
            # using the matmul.
            sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
            sampled_logits = tf.matmul(example_emb,
                                       sampled_w,
                                       transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits


def create_final_embedding(radius, theta):
    """
    return embeddings in cartesian co-ordinates in the poincare disk
    :param radius: A numpy array of shape (n_examples)
    :param theta: A numpy array of shape (n_examples)
    :return: A numpy array of shape (n_examples, 2)
    """
    final_embedding = np.zeros(shape=(len(radius), 2))
    poincare_radius = np.tanh(0.5 * radius)
    final_embedding[:, 0] = poincare_radius * np.cos(theta)
    final_embedding[:, 1] = poincare_radius * np.sin(theta)
    return final_embedding


def main(params):
    """Train a word2vec model in steps of epochs - this is consistent with the tensorflow code."""
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            # Build the graph
            print('initialising model')
            model = cust2vec(params, session)
            # session.run(tf.global_variables_initializer())
        for training_epoch in xrange(params.epochs_to_train):
            print('running epoch {}'.format(training_epoch + 1))
            model.train()  # Process one epoch
        # Perform a final save.
        emb_in, emb_out = model._session.run([model.emb, model.sm_w_t])

    def sort_by_idx(embedding, reverse_index):
        """
        Generate a numpy array with the rows in the same order as the labels
        :param embeddings:
        :param reverse_index:
        :return:
        """
        df = pd.DataFrame(data=embedding, index=np.array(reverse_index))
        sorted_df = df.sort_index()
        return sorted_df.values

    sorted_emb_in = sort_by_idx(emb_in, model._id2word)
    sorted_emb_out = sort_by_idx(emb_out, model._id2word)

    # final_embedding = normalize(final_embedding, norm='l2', axis=0)

    return sorted_emb_in, sorted_emb_out


def karate_results(embeddings, names, n_reps, train_size):
    deepwalk_path = '../../local_resources/karate/size8_walks1_len10.emd'

    y_path = '../../local_resources/karate/y.p'
    x_path = '../../local_resources/karate/X.p'

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
    all_results = utils.merge_results(results, n_reps)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('../../results/karate/tf_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('../../results/karate/tf_micro_pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/karate/tf_macro' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/karate/tf_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)
    return results


def karate_test_scenario(deepwalk_path):
    # deepwalk_path = '../../local_resources/hyperbolic_embeddings/tf_test1.csv'

    y_path = '../../local_resources/karate/y.p'
    x_path = '../../local_resources/karate/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['deepwalk'], ['logistic']]

    x_deepwalk = pd.read_csv(deepwalk_path, index_col=0)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    X = [x_deepwalk.values, normalize(x, axis=0)]
    n_folds = 10
    results = run_detectors.run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('../../results/karate/deepwalk_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('../../results/karate/deepwalk_micro_pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/karate/deepwalk_macro' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/karate/deepwalk_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def karate_embedding_scenario():
    import visualisation
    walk_path = '../../local_resources/karate/walks_n1_l10.csv'
    # walks = pd.read_csv('../../local_resources/karate/walks_n1_l10.csv', header=None).values
    x_path = '../../local_resources/karate/X.p'
    y_path = '../../local_resources/karate/y.p'

    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    # vocab_size = get_vocab_size(x_path, bipartite=False)
    # print('vocab of size: ', vocab_size)
    # # define the noise distribution
    # elems, unigrams = np.unique(walks, return_counts=True)
    # print('unigram distribution', zip(elems, unigrams))
    embeddings = []
    names = []
    n_reps = 10
    train_size = 4  # the number of labelled points to use
    size = 2
    # for window in xrange(1, 10):
    for window in xrange(1, 2):
        params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=3, num_pairs=1500,
                        statistics_interval=2,
                        initial_learning_rate=0.2, epochs=1, concurrent_steps=1)
        embedding, reverse_index = main(params)
        embeddings.append(embedding)
        names.append(['window' + str(window)])
        if size == 2:
            visualisation.plot_embedding(embedding, y, '../../results/karate/figs/window_' + str(
                window) + '_' + utils.get_timestamp() + '.pdf')
    karate_results(embeddings, names, n_reps, train_size)


def generate_karate_embedding():
    import visualisation
    y_path = '../../local_resources/karate/y.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/run4/'
    walk_path = '../../local_resources/karate/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=30, skip_window=5, num_pairs=1500,
                    statistics_interval=0.1,
                    initial_learning_rate=10.0, save_path=log_path, epochs=1, concurrent_steps=1)

    path = '../../local_resources/hyperbolic_embeddings/tf_Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = main(params)
    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/karate/figs/poincare_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/karate/figs/poincare_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=range(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=range(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/hyperbolic_embeddings/tf_Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    return path


if __name__ == '__main__':
    s = datetime.datetime.now()
    path = generate_karate_embedding()
    karate_test_scenario(path)
    # karate_embedding_scenario()

    print(datetime.datetime.now() - s)
