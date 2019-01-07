"""
neural embeddings on the hyperboloid model.
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
        self.embedding_size = embedding_size  # the number of free parameters in the embeddings i.e. the hyperboloid will have embedding_size + 1 dimensions
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

    def transform_grads(self, grad):
        """
        multiply by the inverse of the Minkowski metric tensor g = diag[-1, 1,1 ... 1] to make the first element of each
        grad vector negative
        :param grad: grad matrix of shape (n_vars, embedding_dim)
        :return:
        """
        try:
            x = np.eye(grad.shape[1])
        except IndexError:
            x = np.eye(grad.shape[0])
            grad = tf.expand_dims(grad, 0)
        x[0, 0] = -1.
        T = tf.constant(x, dtype=grad.dtype)
        return tf.matmul(grad, T)

    def rsgd(self, grads, var, lr=1., max_norm=1.):
        """
        Perform the Riemannian gradient descent operation by
        1/ Transforming gradients using the Minkowski metric tensor
        2/ Projecting onto the tangent space
        3/ Applying the exponential map
        :param grads: (grad, name) tuple where grad is a struct with attributes values and indices
        :param var:
        :param lr:
        :return:
        """
        grad, name = grads
        clipped_grads = tf.clip_by_norm(grad.values, max_norm, axes=1)
        # clipped_grads = grad.values
        minkowski_grads = self.transform_grads(clipped_grads)
        # minkowski_grads = self.transform_grads(grad.values)
        vecs = tf.nn.embedding_lookup(var, grad.indices)
        tangent_grads = self.project_tensors_onto_tangent_space(vecs, minkowski_grads)
        return self.tensor_exp_map(var, grad.indices, lr * tangent_grads)
        # self.tensor_exp_map(var, grad.indices, lr * tangent_grads)

    def to_hyperboloid_points(self, vocab_size, embedding_size, init_width):
        """
        Post: result.shape[1] == poincare_pts.shape[1] + 1
        """
        assert np.sqrt(
            embedding_size) * init_width < 1., 'choice of init_width and embedding size allow points to be initialised outside of the poincare ball'
        poincare_pts = np.random.uniform(-init_width, init_width, (vocab_size, embedding_size))
        norm_sqd = (poincare_pts ** 2).sum(axis=1)
        # the hyperboloid has one extra ambient dimension
        result = np.zeros((poincare_pts.shape[0], embedding_size + 1), dtype=np.float64)
        result[:, 1:] = (2. / (1 - norm_sqd))[:, np.newaxis] * poincare_pts
        result[:, 0] = (1 + norm_sqd) / (1 - norm_sqd)
        return result

    def nickel_initialisation(self, vocab_size, embedding_size, init_width=0.001):
        """
        The scheme for initialising points on the hyperboloid used by Nickel and Kiela 18
        :param vocab_size: number of vectors
        :param embedding_size: dimension of each vector
        :param init_width: sample points from (-init_width, init_width) uniformly. 0.001 is the value published by Nickel and Kiela
        :return:
        """
        hyperboloid_points = np.zeros((vocab_size, embedding_size + 1))
        hyperboloid_points[:, 1:] = np.random.uniform(-init_width, init_width,
                                                      size=(vocab_size, embedding_size))
        hyperboloid_points[:, 0] = np.sqrt((hyperboloid_points[:, 1:embedding_size] ** 2).sum(axis=1) + 1)
        return hyperboloid_points

    def to_poincare_ball_points(self, hyperboloid_pts):
        """
        project from hyperboloid to poincare ball
        :param hyperboloid_pts:
        :return:
        """
        N = hyperboloid_pts.shape[1]
        return np.divide(hyperboloid_pts[:, 1:N], hyperboloid_pts[:, 0][:, None] + 1)

    def initialise_on_hyperboloid(self):
        pass

    def minkowski_dot(self, u, v):
        """
        Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
        :param u: a vector
        :param v: a vector
        :return: a scalar dot product
        """
        return tf.tensordot(u, v, 1) - 2 * tf.multiply(u[0], v[0])

    def minkowski_tensor_dot(self, u, v):
        """
        Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
        :param u: a tensor of shape (#examples, dims)
        :param v: a tensor of shape (#examples, dims)
        :return: a scalar dot product
        """
        hyperboloid_dims = self._options.embedding_size + 1
        temp = np.eye(hyperboloid_dims)
        temp[0, 0] = -1.
        T = tf.constant(temp, dtype=u.dtype)
        # make the first column of v negative
        v_neg = tf.matmul(v, T)
        return tf.reduce_sum(tf.multiply(u, v_neg), 1, keep_dims=True)  # keep dims for broadcasting

    def pairwise_minkowski_dot(self, u, v):
        """
        creates a matrix of minkowski dot products M(i,j) = u[i,:]*v[j,:]
        :param examples: first set of vectors of shape (ndata1, ndim)
        :param samples: second set of vectors of shape (ndata2, ndim)
        :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
        """
        try:
            temp = np.eye(u.shape[1])
        except IndexError:
            temp = np.eye(u.shape)
        temp[0, 0] = -1.
        T = tf.constant(temp, dtype=u.dtype)
        # make the first column of v negative
        v_neg = tf.matmul(v, T)
        return tf.matmul(u, v_neg, transpose_b=True)

    def minkowski_dist(self, u, v):
        """
        The distance between points in Minkowski space
        :param u: tensor of points of shape (examples, dims)
        :param v: tensor of points of shape (examples, dims)
        :return: a tensor of distances of shape (examples)
        """
        return tf.acosh(-tf.minimum(self.minkowski_tensor_dot(u, v), -1.))

    def project_onto_tangent_space(self, hyperboloid_point, minkowski_tangent):
        """
        project gradients in the ambiant space onto the tangent space
        :param hyperboloid_point:
        :param minkowski_tangent:
        :return:
        """
        return minkowski_tangent + self.minkowski_dot(hyperboloid_point, minkowski_tangent) * hyperboloid_point

    def project_tensors_onto_tangent_space(self, hyperboloid_points, ambient_gradients):
        """
        project gradients in the ambiant space onto the tangent space
        :param hyperboloid_point: A point on the hyperboloid
        :param ambient_gradient: The gradient to project
        :return: gradients in the tangent spaces of the hyperboloid points
        """
        return ambient_gradients + tf.multiply(self.minkowski_tensor_dot(hyperboloid_points, ambient_gradients),
                                               hyperboloid_points)

    def exp_map(self, base, tangent):
        """
        Map a vector 'tangent' from the tangent space at point 'base' onto the manifold.
        """
        # tangent = tangent.copy()
        norm = tf.sqrt(tf.maximum(self.minkowski_dot(tangent, tangent), 0))
        if norm == 0:
            return base
        tangent /= norm
        return tf.cosh(norm) * base + tf.sinh(norm) * tangent

    # def tensor_exp_map(self, vars, indices, tangent_grads):
    #     """
    #     Map vectors in the tangent space of the hyperboloid points back onto the hyperboloid
    #     :param hyperboloid_points: a tensor of points on the hyperboloid of shape (#examples, #dims)
    #     :param tangent_grads: a tensor of gradients on the tangent spaces of the hyperboloid_points of shape (#examples, #dims)
    #     :return:
    #     """
    #     # todo do we need to normalise the gradients?
    #     hyperboloid_points = tf.nn.embedding_lookup(vars, indices)
    #     embedding_size = self._options.embedding_size
    #     batch_size = self._options.batch_size
    #     # set shape is required as boolean mask can not use tensors of unknown shape
    #     tangent_grads.set_shape([batch_size, embedding_size + 1])
    #     norms = tf.sqrt(tf.maximum(self.minkowski_tensor_dot(tangent_grads, tangent_grads), 0))
    #     # norms.set_shape([batch_size, 1])
    #     norms.set_shape([None, 1])
    #     zero = tf.constant(0, dtype=tf.float32)
    #     nonzero_flags = tf.squeeze(tf.not_equal(norms, zero))
    #     # nonzero_flags = tf.not_equal(norms, zero)
    #     # nonzero_flags.set_shape([batch_size, 1])
    #     nonzero_flags.set_shape([None])
    #     nonzero_indices = tf.boolean_mask(indices, nonzero_flags)
    #     print('norms shape: ', norms.shape)
    #     print('nonzero_flags shape: ', nonzero_flags.shape)
    #     print('tangent grads shape: ', tangent_grads.shape)
    #     nonzero_norms = tf.boolean_mask(norms, nonzero_flags)
    #     updated_grads = tf.boolean_mask(tangent_grads, nonzero_flags)
    #     updated_points = tf.boolean_mask(hyperboloid_points, nonzero_flags)
    #     normed_grads = tf.divide(updated_grads, nonzero_norms)
    #     updates = tf.multiply(tf.cosh(nonzero_norms), updated_points) + tf.multiply(tf.sinh(nonzero_norms),
    #                                                                                 normed_grads)
    #     return tf.scatter_update(vars, nonzero_indices, updates)

    def tensor_exp_map(self, vars, indices, tangent_grads):
        """
        Map vectors in the tangent space of the hyperboloid points back onto the hyperboloid
        :param hyperboloid_points: a tensor of points on the hyperboloid of shape (#examples, #dims)
        :param tangent_grads: a tensor of gradients on the tangent spaces of the hyperboloid_points of shape (#examples, #dims)
        :return:
        """
        # todo do we need to normalise the gradients?
        hyperboloid_points = tf.nn.embedding_lookup(vars, indices)
        norms = tf.sqrt(tf.maximum(self.minkowski_tensor_dot(tangent_grads, tangent_grads), 0))
        normed_grads = tf.divide(tangent_grads, norms)  # norms can be zero, so may contain nan and inf values
        updates = tf.multiply(tf.cosh(norms), hyperboloid_points) + tf.multiply(tf.sinh(norms),
                                                                                normed_grads)  # norms can also be large sending sinh / cosh -> inf
        values_to_replace = tf.logical_or(tf.is_nan(updates), tf.is_inf(updates))
        safe_updates = tf.where(values_to_replace, hyperboloid_points, updates)
        safe_updates = self.project_onto_manifold(safe_updates)
        return tf.scatter_update(vars, indices, safe_updates)

    def pairwise_distance(self, examples, samples):
        """
        creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
        :param examples: first set of vectors of shape (ndata1, ndim)
        :param samples: second set of vectors of shape (ndata2, ndim)
        :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
        """
        dist = tf.acosh(-tf.minimum(self.pairwise_minkowski_dot(examples, samples), -1.))
        return dist

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

    # def project_onto_manifold(self, tensor):
    #     """
    #     project a tensor back onto the hyperboloid by fixing the first coordinate to sqrt(|x[1:]|^2 + 1)
    #     :param tensor: a tensor of shape (examples, dimensions) where dimensions > 2
    #     :return:
    #     """
    #     norm_square = tf.square(tensor[:, 1:])
    #     first_coord = tf.sqrt(tf.reduce_sum(norm_square, axis=1) + 1)
    #     return tensor[:, 0].assign(first_coord)

    def project_onto_manifold(self, tensor):
        """
        project a tensor back onto the hyperboloid by fixing the first coordinate to sqrt(|x[1:]|^2 + 1)
        :param tensor: a tensor of shape (examples, dimensions) where dimensions > 2
        :return:
        """
        kept_values = tensor[:, 1:]
        norm_square = tf.square(kept_values)
        new_vals = tf.expand_dims(tf.sqrt(tf.reduce_sum(norm_square, axis=1) + 1), axis=1)
        return tf.concat([new_vals, kept_values], axis=1)

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

            grads = optimizer.compute_gradients(loss, [self.sm_b, self.emb, self.sm_w_t])
            sm_b_grad, emb_grad, sm_w_t_grad = [(self.remove_nan(grad), var) for grad, var in grads]

            sm_b_grad_hist = tf.summary.histogram('smb_grad', sm_b_grad[0])
            emb_grad_hist = tf.summary.histogram('emb_grad', emb_grad[0])
            sm_w_t_grad_hist = tf.summary.histogram('sm_w_t_grad', sm_w_t_grad[0])

            self.emb_grad = emb_grad
            self.sm_w_t_grad = sm_w_t_grad
            self.sm_b_grad = sm_b_grad

            # gv = [sm_b_grad, emb_grad, sm_w_t_grad]
            # vars = [self.sm_b, self.emb, self.sm_w_t]
            gv = [emb_grad, sm_w_t_grad]
            vars = [self.emb, self.sm_w_t]

            all_update_ops = [optimizer.apply_gradients([self.sm_b_grad], global_step=self.global_step)]
            for var, grad in zip(vars, gv):
                # get riemannian factor
                # rescale grad
                # all_update_ops.append(tf.assign(var, self.rsgd(grad, var, lr)))
                all_update_ops.append(self.rsgd(grad, var, lr))
                # all_update_ops.append(self.rsgd(grad, var, lr))

            self._train = tf.group(*all_update_ops)

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
        self.true_logits = true_logits
        self.sampled_logits = sampled_logits
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar("NCE_loss", loss)
        self._loss = loss
        self.optimize(loss)
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
            (epoch, step, loss, words, lr, examples, labels, samples, true_logits, sampled_logits, emb, smw_t, sm_b,
             emb_grads,
             smwt_grads, dist, example_emb, sampled_w) = self._session.run(
                [self._epoch, self.global_step, self._loss, self._words, self._lr, self.examples,
                 self.labels, self.samples, self.true_logits, self.sampled_logits, self.emb, self.sm_w_t, self.sm_b,
                 self.emb_grad, self.sm_w_t_grad, self.dist, self.example_emb, self.sampled_w])
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
            if np.isnan(loss):
                print('true logits: ', true_logits)
                print('sampled logits: ', sampled_logits)
                print("labels: ", labels, " examples: ", examples, "samples: ", samples)
                print('example embedding: ', emb[examples, :])
                print('label embedding: ', smw_t[labels, :])
                print('sample embedding: ', smw_t[samples, :])
                print('sm_b: ', sm_b)
                print('dist: ', dist)
                print('example emb: ', example_emb)
                print('sample_w: ', sampled_w)

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
        # use the expanded version of the l2 norm to simplify broadcasting ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy.T
        euclidean_dist_sq = xnorm_sq[:, None] + ynorm_sq[None, :] - 2 * tf.matmul(examples, samples, transpose_b=True)
        denom = (1 - xnorm_sq[:, None]) * (1 - ynorm_sq[None, :])
        hyp_dist = tf.acosh(1 + 2 * tf.divide(euclidean_dist_sq, denom))
        return hyp_dist

    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        # Embedding: [vocab_size, emb_dim]
        opts = self._options
        with tf.name_scope('model'):
            init_width = 0.5 / (1 * opts.embedding_size)

            self.emb = tf.Variable(self.nickel_initialisation(opts.vocab_size, opts.embedding_size, init_width),
                                   name="emb", dtype=tf.float32)

            emb_hist = tf.summary.histogram('embedding', self.emb)

            self.sm_w_t = tf.Variable(self.nickel_initialisation(opts.vocab_size, opts.embedding_size, init_width),
                                      name="sm_w_t", dtype=tf.float32)

            smw_hist = tf.summary.histogram('softmax_weight', self.sm_w_t)

            # Softmax bias: [emb_dim].
            self.sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
            smb_hist = tf.summary.histogram('softmax_bias', self.sm_b)

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
            self.example_emb = example_emb
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

            self.samples = sampled_ids

            # Weights for sampled ids: [num_sampled, emb_dim]
            sampled_w = tf.nn.embedding_lookup(self.sm_w_t, sampled_ids)
            self.sampled_w = sampled_w

            # sampled_w = self.clip_indexed_slices_norms(unorm_sampled_w)
            # Biases for sampled ids: [num_sampled, 1]
            sampled_b = tf.nn.embedding_lookup(self.sm_b, sampled_ids)

            # True logits: [batch_size, 1]
            # true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
            true_logits = self.minkowski_dist(example_emb, true_w) + true_b
            print('true logits shape: ', true_logits.shape)

            # Sampled logits: [batch_size, num_sampled]
            # We replicate sampled noise labels for all examples in the batch
            sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])

            dist = self.pairwise_distance(example_emb, sampled_w)
            self.dist = dist
            sampled_logits = dist + sampled_b_vec
            print('sampled logits shape: ', sampled_logits.shape)
        return true_logits, sampled_logits


def main(params, path1, path2):
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
        hyperboloid_emb_in, hyperboloid_emb_out = model._session.run([model.emb, model.sm_w_t])
        df_hyp_in = pd.DataFrame(data=hyperboloid_emb_in, index=range(hyperboloid_emb_in.shape[0]))
        df_hyp_in.to_csv(path1, sep=',')
        df_hyp_out = pd.DataFrame(data=hyperboloid_emb_out, index=range(hyperboloid_emb_out.shape[0]))
        df_hyp_out.to_csv(path2, sep=',')
        emb_in = model.to_poincare_ball_points(hyperboloid_emb_in)
        emb_out = model.to_poincare_ball_points(hyperboloid_emb_out)

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
                    initial_learning_rate=1., save_path=log_path, epochs=1, concurrent_steps=1)

    path = '../../local_resources/karate/embeddings/lorentzian_Win' + '_' + utils.get_timestamp() + '.csv'
    hyp_path_in = '../../local_resources/karate/embeddings/lorentzian_hyp_Win' + '_' + utils.get_timestamp() + '.csv'
    hyp_path_out = '../../local_resources/karate/embeddings/lorentzian_hyp_Wout' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = main(params, hyp_path_in, hyp_path_out)
    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/karate/figs/lorentzian_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/karate/figs/lorentzian_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=range(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=range(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/karate/embeddings/lorentzian_Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    return path


if __name__ == '__main__':
    s = datetime.datetime.now()
    path = generate_karate_embedding()
    karate_test_scenario(path)

    print(datetime.datetime.now() - s)
