"""
neural embeddings on the hyperboloid model
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
from sklearn.linear_model import LogisticRegression
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
        # scaled_grads = []
        grad, name = grads
        vecs = tf.nn.embedding_lookup(emb, grad.indices)
        norm_squared = tf.square(tf.norm(vecs, axis=0))
        hyperbolic_factor = 0.25 * tf.square(1 - norm_squared)
        g = tf.multiply(grad.values, hyperbolic_factor)
        scaled_grad = tf.IndexedSlices(g, grad.indices, grad.dense_shape)
        return scaled_grad, name

    def clip_tensor_norms(self, emb, epsilon=0.00001):
        """
        not used as clip_by_norm performs this task
        :param epsilon:
        :return:
        """
        norms = tf.norm(emb, axis=1)
        comparison = tf.greater_equal(norms, tf.constant(1.0, dtype=tf.float32))
        norm = tf.nn.l2_normalize(emb, dim=1) - epsilon
        conditional_assignment_op = emb.assign(tf.where(comparison, norm, emb))
        return conditional_assignment_op

    def clip_indexed_slices_norms(self, emb_slice, epsilon=0.00001):
        """
        not used as clip_by_norm performs this task
        :param epsilon:
        :return:
        """
        norms = tf.norm(emb_slice, axis=1)
        comparison = tf.greater_equal(norms, tf.constant(1.0, dtype=tf.float32))
        norm = tf.nn.l2_normalize(emb_slice, dim=1) - epsilon
        normed_slice = tf.where(comparison, norm, emb_slice)
        return normed_slice

    def clip_norms(self, epsilon=0.00001):
        """
        not used as clip_by_norm performs this task
        :param epsilon:
        :return:
        """
        emb_norms = tf.norm(self.emb, axis=1)
        sm_norms = tf.norm(self.sm_w_t, axis=1)
        emb_comparison = tf.greater_equal(emb_norms, tf.constant(1.0, dtype=tf.float32))
        sm_comparison = tf.greater_equal(sm_norms, tf.constant(1.0, dtype=tf.float32))
        emb_norms = tf.nn.l2_normalize(self.emb, dim=1) - epsilon
        sm_norms = tf.nn.l2_normalize(self.sm_w_t, dim=1) - epsilon
        self._clip_emb = self.emb.assign(tf.where(emb_comparison, emb_norms, self.emb))
        self._clip_sm = self.sm_w_t.assign(tf.where(sm_comparison, sm_norms, self.sm_w_t))

    def remove_nan(self, grad, epsilon=0.001):
        """
        replace nans in gradients caused by comparing two indentical vectors with a small gradient
        :param grad: a tf.IndexedSlicesValue
        :param epsilon: the value to set nan values to
        :return: the tensor with nans replaced by epsilon
        """
        g1 = tf.where(tf.is_nan(grad.values), tf.ones_like(grad.values) * epsilon, grad.values)
        g2 = tf.where(tf.is_inf(g1), tf.ones_like(g1) * epsilon, g1)
        safe_grad = tf.IndexedSlices(g2, grad.indices, grad.dense_shape)
        return safe_grad

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        with tf.name_scope('optimize'):
            epsilon = 1e-5
            opts = self._options
            words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
            lr = opts.learning_rate * tf.maximum(
                0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
            self._lr = lr
            optimizer = tf.train.GradientDescentOptimizer(lr)
            # self.emb = tf.clip_by_norm(self.emb, 1 - epsilon, axes=1)
            # self.sm_w_t = tf.clip_by_norm(self.sm_w_t, 1 - epsilon, axes=1)
            # clip the vectors back inside the Poincare ball
            self.clip_tensor_norms(self.emb)
            self.clip_tensor_norms(self.sm_w_t)
            grads = optimizer.compute_gradients(loss, [self.sm_b, self.emb, self.sm_w_t])
            sm_b_grad, emb_grad, sm_w_t_grad = [(self.remove_nan(grad), var) for grad, var in grads]

            # emb_grad = optimizer.compute_gradients(loss, [self.emb])
            # sm_w_t_grad = optimizer.compute_gradients(loss, [self.sm_w_t])

            sm_b_grad_hist = tf.summary.histogram('smb_grad', sm_b_grad[0])
            emb_grad_hist = tf.summary.histogram('emb_grad', emb_grad[0])
            sm_w_t_grad_hist = tf.summary.histogram('sm_w_t_grad', sm_w_t_grad[0])

            self.emb_grad = emb_grad
            self.sm_w_t_grad = sm_w_t_grad

            # modified_emb_grad = emb_grad
            # modified_sm_w_t_grad = sm_w_t_grad
            # modified_sm_b_grad = self.modify_grads(sm_b_grad, self.sm_b)
            modified_emb_grad = self.modify_grads(emb_grad, self.emb)
            modified_sm_w_t_grad = self.modify_grads(sm_w_t_grad, self.sm_w_t)
            # theta_out_clipped = tf.clip_by_value(modified_theta_out, -1, 1, name="theta_out_clipped")
            self.modified_emb_grad = modified_emb_grad
            self.modified_sm_w_t_grad = modified_sm_w_t_grad

            modified_emb_grad_hist = tf.summary.histogram('modified_emb_grad', modified_emb_grad[0])
            modified_sm_w_t_grad_hist = tf.summary.histogram('modified_sm_w_t_grad', modified_sm_w_t_grad[0])

            gv = [sm_b_grad, modified_emb_grad, modified_sm_w_t_grad]

            # gv = sm_b_grad + emb_grad + sm_w_t_grad
            # gv = [sm_b_grad, modified_emb_grad, modified_sm_w_t_grad]
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
        # add operator to clip embeddings inside the poincare ball
        self.clip_norms()
        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        # Add opp to save variables
        self.saver = tf.train.Saver()

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        # print('thread sees initial epoch: ', initial_epoch)
        while True:
            # this line increases the runtime by 1000 times
            # self._session.run([self._clip_emb, self._clip_sm])
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

    def elementwise_distance(self, examples, labels):
        """
        creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
        :param examples: first set of vectors of shape (ndata1, ndim)
        :param labels: second set of vectors of shape (ndata2, ndim)
        :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
        """
        xnorm_sq = tf.reduce_sum(tf.square(examples), axis=1)
        ynorm_sq = tf.reduce_sum(tf.square(labels), axis=1)
        euclidean_dist_sq = tf.reduce_sum(tf.square(examples - labels), axis=1)
        denom = tf.multiply(1 - xnorm_sq, 1 - ynorm_sq)
        hyp_dist = tf.acosh(1 + 2 * tf.divide(euclidean_dist_sq, denom))
        return hyp_dist

    def pairwise_distance(self, examples, samples):
        """
        creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
        :param examples: first set of vectors of shape (ndata1, ndim)
        :param samples: second set of vectors of shape (ndata2, ndim)
        :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
        """
        xnorm_sq = tf.reduce_sum(tf.square(examples), axis=1)
        ynorm_sq = tf.reduce_sum(tf.square(samples), axis=1)
        # use the multiplied out version of the l2 norm to simplify broadcasting ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy.T
        euclidean_dist_sq = xnorm_sq[:, None] + ynorm_sq[None, :] - 2 * tf.matmul(examples, samples, transpose_b=True)
        denom = (1 - xnorm_sq[:, None]) * (1 - ynorm_sq[None, :])
        hyp_dist = tf.acosh(1 + 2 * tf.divide(euclidean_dist_sq, denom))
        return hyp_dist

    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        # Embedding: [vocab_size, emb_dim]
        opts = self._options
        with tf.name_scope('model'):
            init_width = 0.5 / opts.embedding_size
            # emb = np.random.uniform(low=-init_width, high=init_width,
            #                         size=(opts.vocab_size, opts.embedding_size)).astype(np.float32)

            self.emb = tf.Variable(
                tf.random_uniform(
                    [opts.vocab_size, opts.embedding_size], -init_width, init_width),
                name="emb")

            emb_hist = tf.summary.histogram('embedding', self.emb)

            # Softmax weight: [vocab_size, emb_dim]. Transposed.
            # self.sm_w_t = tf.Variable(
            #     tf.zeros([opts.vocab_size, opts.embedding_size]),
            #     name="sm_w_t")

            self.sm_w_t = tf.Variable(
                tf.random_uniform(
                    [opts.vocab_size, opts.embedding_size], -init_width, init_width),
                name="sm_w_t")

            smw_hist = tf.summary.histogram('softmax weight', self.sm_w_t)

            # Softmax bias: [emb_dim].
            self.sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
            smb_hist = tf.summary.histogram('softmax bias', self.sm_b)

            # Create a variable to keep track of the number of batches that have been fed to the graph
            self.global_step = tf.Variable(0, name="global_step")

            # with tf.name_scope('input'):
            # Nodes to compute the nce loss w/ candidate sampling.
            labels_matrix = tf.reshape(
                tf.cast(labels,
                        dtype=tf.int64),
                [opts.batch_size, 1])

            # norm_emb = self.clip_tensor_norms(self.emb)
            # norm_sm_w_t = self.clip_tensor_norms(self.sm_w_t)

            # Embeddings for examples: [batch_size, emb_dim]
            example_emb = tf.nn.embedding_lookup(self.emb, examples)
            # example_emb = self.clip_indexed_slices_norms(unorm_emb)
            # example_hist = tf.summary.histogram('input embeddings', example_emb)

            # Weights for labels: [batch_size, emb_dim]
            true_w = tf.nn.embedding_lookup(self.sm_w_t, labels)
            # true_w = self.clip_indexed_slices_norms(unorm_true_w)

            # Biases for labels: [batch_size, 1]
            true_b = tf.nn.embedding_lookup(self.sm_b, labels)

            # with tf.name_scope('negative_samples'):
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
            # sampled_w = self.clip_indexed_slices_norms(unorm_sampled_w)
            # Biases for sampled ids: [num_sampled, 1]
            sampled_b = tf.nn.embedding_lookup(self.sm_b, sampled_ids)

            # True logits: [batch_size, 1]
            # true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
            true_logits = self.elementwise_distance(example_emb, true_w) + true_b

            # Sampled logits: [batch_size, num_sampled]
            # We replicate sampled noise labels for all examples in the batch
            # using the matmul.
            sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
            # sampled_logits = tf.matmul(example_emb,
            #                            sampled_w,
            #                            transpose_b=True) + sampled_b_vec
            sampled_logits = self.pairwise_distance(example_emb, sampled_w) + sampled_b_vec
        return true_logits, sampled_logits


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


def karate_test_scenario(deepwalk_path):
    y_path = '../../local_resources/karate/y.p'
    x_path = '../../local_resources/karate/X.p'

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['deepwalk'], ['logistic']]

    x_deepwalk = pd.read_csv(deepwalk_path, index_col=0)
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


def generate_karate_embedding():
    import visualisation
    y_path = '../../local_resources/karate/y.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/hyperbolic_cartesian/lr1_epoch1_dim4'
    walk_path = '../../local_resources/karate/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=0.001,
                    initial_learning_rate=0.2, save_path=log_path, epochs=5, concurrent_steps=1)

    path = '../../local_resources/karate/embeddings/hyperbolic_cartesian_Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = main(params)
    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/karate/figs/poincare_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/karate/figs/poincare_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=range(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=range(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/karate/embeddings/hyperbolic_cartesian_Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    return path


if __name__ == '__main__':
    s = datetime.datetime.now()
    path = generate_karate_embedding()
    karate_test_scenario(path)

    print(datetime.datetime.now() - s)
