"""
Tests for calculating lorentzian embeddings in the hyperboloid with tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import datetime

# from lorentzian_embedding import cust2vec, Params

sys.path.append(os.path.join('..', 'src', 'python'))

import numpy as np
import pandas as pd
import tensorflow as tf
import math
import utils


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


class TestClass:
    def __init__(self, unigrams, params):
        self.batch_size = params.batch_size
        self.embedding_size = params.embedding_size
        self.vocab_size = len(unigrams)
        self.unigrams = unigrams
        self.num_samples = params.num_samples
        self.lr = params.learning_rate
        self.build_graph()

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
        vecs = tf.nn.embedding_lookup(var, grad.indices)
        tangent_grads = self.project_tensors_onto_tangent_space(vecs, minkowski_grads)
        return self.tensor_exp_map(var, grad.indices, lr * tangent_grads)

    def project_tensors_onto_tangent_space(self, hyperboloid_points, ambient_gradients):
        """
        project gradients in the ambiant space onto the tangent space
        :param hyperboloid_point: A point on the hyperboloid
        :param ambient_gradient: The gradient to project
        :return: gradients in the tangent spaces of the hyperboloid points
        """
        return ambient_gradients + tf.multiply(self.minkowski_tensor_dot(hyperboloid_points, ambient_gradients),
                                               hyperboloid_points)

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

    def forward(self, examples, labels):
        init_width = 0.5 / self.embedding_size
        # self.emb = tf.Variable(self.to_hyperboloid_points(self.vocab_size, self.embedding_size, init_width),
        #                        name="emb", dtype=tf.float32)
        #
        # self.sm_w_t = tf.Variable(self.to_hyperboloid_points(self.vocab_size, self.embedding_size, init_width),
        #                           name="sm_w_t", dtype=tf.float32)

        self.emb = tf.Variable(self.nickel_initialisation(self.vocab_size, self.embedding_size, init_width),
                               name="emb", dtype=tf.float32)

        self.sm_w_t = tf.Variable(self.nickel_initialisation(self.vocab_size, self.embedding_size, init_width),
                                  name="sm_w_t", dtype=tf.float32)

        self.sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")
        # print('labels shape: ', labels.shape)
        # print('examples shape: ', examples.shape)
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

        self.samples = sampled_ids

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(self.emb, examples)
        self.example_emb = example_emb

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(self.sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(self.sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(self.sm_w_t, sampled_ids)
        self.sampled_w = sampled_w
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(self.sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        # true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
        # print('example shape: ', example_emb.shape)
        # print('true_w shape: ', true_w.shape)
        dist = self.minkowski_dist(example_emb, true_w)
        true_logits = dist + true_b
        self.dist = dist
        self.true_logits = true_logits

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.num_samples])

        pairwise_dist = self.pairwise_distance(example_emb, sampled_w)
        self.pairwise_dist = pairwise_dist
        sampled_logits = dist + sampled_b_vec
        # sampled_logits = self.pairwise_distance(example_emb, sampled_w) + sampled_b_vec
        self.sampled_logits = sampled_logits
        # sampled_logits = tf.matmul(example_emb,
        #                            sampled_w,
        #                            transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""
        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size
        return nce_loss_tensor

    def pairwise_distance(self, examples, samples):
        """
        creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
        :param examples: first set of vectors of shape (ndata1, ndim)
        :param samples: second set of vectors of shape (ndata2, ndim)
        :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
        """
        return tf.acosh(-tf.minimum(self.pairwise_minkowski_dot(examples, samples), -1.))

    def minkowski_tensor_dot(self, u, v):
        """
        Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
        :param u: a tensor of shape (#examples, dims)
        :param v: a tensor of shape (#examples, dims)
        :return: a scalar dot product
        """
        # assert u.shape == v.shape, 'minkowski dot product not define for u of shape {} and v of shape'.format(u.shape,                                                                                                              v.shape)
        hyperboloid_dims = self.embedding_size + 1
        # try:
        #     temp = np.eye(u.shape[1])
        # except IndexError:
        #     temp = np.eye(u.shape)
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

    def build_graph(self):
        self.examples = tf.placeholder(tf.int32, shape=[self.batch_size], name='examples')
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_size], name='labels')
        true_logits, sampled_logits = self.forward(self.examples, self.labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        self._loss = loss
        self.optimize(loss)

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

        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        # grads = optimizer.compute_gradients(loss, [self.sm_b, self.emb, self.sm_w_t])
        grads = optimizer.compute_gradients(loss, [self.sm_b, self.emb, self.sm_w_t])
        sm_b_grad, emb_grad, sm_w_t_grad = [(self.remove_nan(grad), var) for grad, var in grads]

        self.emb_grad = emb_grad
        self.sm_w_t_grad = sm_w_t_grad
        self.sm_b_grad = sm_b_grad

        # gv = [sm_b_grad, emb_grad, sm_w_t_grad]
        # vars = [self.sm_b, self.emb, self.sm_w_t]
        gv = [emb_grad, sm_w_t_grad]
        vars = [self.emb, self.sm_w_t]

        all_update_ops = [optimizer.apply_gradients([self.sm_b_grad])]
        for var, grad in zip(vars, gv):
            # get riemannian factor
            # rescale grad
            # all_update_ops.append(tf.assign(var, self.rsgd(grad, var, lr)))
            all_update_ops.append(self.rsgd(grad, var, self.lr))
            # all_update_ops.append(self.rsgd(grad, var, lr))

        self._train = tf.group(*all_update_ops)


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


def test_loss():
    walk_path = '../local_resources/karate/walks_n1_l10.csv'
    walks = pd.read_csv(walk_path, header=None).values
    elems, unigrams = np.unique(walks, return_counts=True)
    log_path = '.'

    params = Params(walk_path, batch_size=2, embedding_size=2, neg_samples=2, skip_window=5, num_pairs=400,
                    statistics_interval=10,
                    initial_learning_rate=1., save_path=log_path, epochs=5, concurrent_steps=1)
    # initialise the graph
    graph = tf.Graph()
    # run the tensorflow session
    with tf.Session(graph=graph) as session:
        # Define the training data
        model = TestClass(unigrams, params)

        # initialize all variables in parallel
        tf.global_variables_initializer().run()
        _ = [print(v) for v in tf.global_variables()]

        s = datetime.datetime.now()
        print("Initialized")
        # define batch generator
        batch_gen = generate_batch(params.skip_window, walks, params.batch_size)
        average_loss = 0
        n_pairs = 0
        num_steps = params.num_pairs / params.batch_size
        print('running for ', num_steps, ' steps')
        for step in xrange(int(num_steps)):
            s_batch = datetime.datetime.now()
            batch_inputs, batch_labels = batch_gen.next()
            # print('examples: ', batch_inputs)
            feed_dict = {model.examples: batch_inputs, model.labels: np.squeeze(batch_labels)}

            # _, loss_val = session.run([model._train, model._loss], feed_dict=feed_dict)

            (_, loss_val, examples, labels, samples, true_logits, sampled_logits, emb, smw_t, sm_b,
             dist, pairwise_dist, example_emb, sampled_w, sm_b_grad) \
                = session.run([model._train, model._loss, model.examples, model.labels, model.samples,
                               model.true_logits, model.sampled_logits, model.emb, model.sm_w_t, model.sm_b,
                               model.dist, model.pairwise_dist, model.example_emb, model.sampled_w, model.sm_b_grad],
                              feed_dict=feed_dict)

            if np.isnan(loss_val):
                print('true logits: ', true_logits)
                print('sampled logits: ', sampled_logits)
                print("labels: ", labels, " examples: ", examples, "samples: ", samples)
                print('example embedding: ', emb[examples, :])
                print('label embedding: ', smw_t[labels, :])
                print('sample embedding: ', smw_t[samples, :])
                print('sm_b: ', sm_b)
                print('sm_b grads: ', sm_b_grad)
                print('dist: ', dist)
                print('pairwise dist: ', pairwise_dist)
                print('example emb: ', example_emb)
                print('sample_w: ', sampled_w)

            average_loss += loss_val
            n_pairs += params.batch_size
            if step % params.statistics_interval == 0:
                if step > 0:
                    average_loss /= params.statistics_interval
                # The average loss is an estimate of the loss over the last 2000 batches.
                runtime = datetime.datetime.now() - s_batch
                print("Average loss at step ", step, ": ", average_loss, 'ran in', runtime)
                s_batch = datetime.datetime.now()
                average_loss = 0
        # final_embeddings = normalized_embeddings.eval()
        final_embedding = model.emb.eval()
        print('ran in {0} s'.format(datetime.datetime.now() - s))
        df_final_embedding = pd.DataFrame(data=final_embedding, index=range(final_embedding.shape[0]))
        path = '../local_resources/karate/embeddings/test_lorentzian_hyp_Win' + '_' + utils.get_timestamp() + '.csv'
        df_final_embedding.to_csv(path, sep=',')


def minkowski_tensor_dot(u, v):
    """
    Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
    :param u: a tensor of shape (#examples, dims)
    :param v: a tensor of shape (#examples, dims)
    :return: a scalar dot product
    """
    assert u.shape == v.shape, 'minkowski dot product not defined for different shape tensors'
    try:
        temp = np.eye(u.shape[1])
    except IndexError:
        temp = np.eye(u.shape)
    temp[0, 0] = -1.
    T = tf.constant(temp, dtype=u.dtype)
    # make the first column of v negative
    v_neg = tf.matmul(v, T)
    return tf.reduce_sum(tf.multiply(u, v_neg), 1, keep_dims=True)  # keep dims for broadcasting


def minkowski_numpy_dot(u, v):
    """
    Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
    :param u: a tensor of shape (#examples, dims)
    :param v: a tensor of shape (#examples, dims)
    :return: a scalar dot product
    """
    assert u.shape == v.shape, 'minkowski dot product not defined for different shape tensors'
    try:
        temp = np.eye(u.shape[1])
    except IndexError:
        temp = np.eye(u.shape)
    temp[0, 0] = -1.
    # make the first column of v negative
    v_neg = np.matmul(v, temp)
    return np.sum(np.multiply(u, v_neg), axis=1)  # keep dims for broadcasting


def pairwise_minkowski_dot(u, v):
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


def minkowski_vector_dot(u, v):
    """
        Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
        :param u: a vector
        :param v: a vector
        :return: a scalar dot product
        """
    assert u.shape == v.shape, 'minkowski dot product not define for different shape vectors'
    # assert that the vectors have only 1d.
    # todo this currently fails because exp_map returns tensors with shape = None
    # assert u.get_shape().ndims == 1, 'applied minkowski_vector_dot to a tensor. Try using minkowski_tensor_dot'

    return tf.tensordot(u, v, 1) - 2 * tf.multiply(u[0], v[0])


def minkowski_dist(u, v):
    """
    The distance between points in Minkowski space
    :param u: tensor of points of shape (examples, dims)
    :param v: tensor of points of shape (examples, dims)
    :return: a tensor of distances of shape (examples)
    """
    return tf.acosh(-tf.minimum(minkowski_tensor_dot(u, v), -1.))


def pairwise_minkowski_dist(u, v):
    """
    creates a matrix of minkowski dot products D(i,j) = d(u[i,:],v[j,:])
    :param examples: first set of vectors of shape (ndata1, ndim)
    :param samples: second set of vectors of shape (ndata2, ndim)
    :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
    """
    return tf.acosh(-tf.minimum(pairwise_minkowski_dot(u, v), -1.))


def project_onto_tangent_space(hyperboloid_point, ambient_gradient):
    """
    project gradients in the ambiant space onto the tangent space
    :param hyperboloid_point: A point on the hyperboloid
    :param ambient_gradient: The gradient to project
    :return:
    """
    return ambient_gradient + minkowski_vector_dot(hyperboloid_point, ambient_gradient) * hyperboloid_point


def project_tensors_onto_tangent_space(hyperboloid_points, ambient_gradients):
    """
    project gradients in the ambiant space onto the tangent space
    :param hyperboloid_point: A point on the hyperboloid
    :param ambient_gradient: The gradient to project
    :return: gradients in the tangent spaces of the hyperboloid points
    """
    return ambient_gradients + tf.multiply(minkowski_tensor_dot(hyperboloid_points, ambient_gradients),
                                           hyperboloid_points)


def exp_map(base, tangent):
    """
    Map a vector 'tangent' from the tangent space at point 'base' onto the manifold.
    """
    norm = tf.sqrt(tf.maximum(minkowski_vector_dot(tangent, tangent), 0))
    if norm == 0:
        return base
    tangent /= norm
    return tf.cosh(norm) * base + tf.sinh(norm) * tangent


def tensor_exp_map1(vars, indices, tangent_grads):
    """
    Map vectors in the tangent space of the hyperboloid points back onto the hyperboloid
    :param hyperboloid_points: a tensor of points on the hyperboloid of shape (#examples, #dims)
    :param tangent_grads: a tensor of gradients on the tangent spaces of the hyperboloid_points of shape (#examples, #dims)
    :return:
    """
    # todo do we need to normalise the gradients?
    hyperboloid_points = tf.nn.embedding_lookup(vars, indices)
    norms = tf.sqrt(tf.maximum(minkowski_tensor_dot(tangent_grads, tangent_grads), 0))
    zero = tf.constant(0, dtype=tf.float32)
    nonzero_flags = tf.squeeze(tf.not_equal(norms, zero))
    nonzero_indices = tf.boolean_mask(indices, nonzero_flags)
    nonzero_norms = tf.boolean_mask(norms, nonzero_flags)
    updated_grads = tf.boolean_mask(tangent_grads, tf.squeeze(nonzero_flags))
    updated_points = tf.boolean_mask(hyperboloid_points, nonzero_flags)
    normed_grads = tf.divide(updated_grads, nonzero_norms)
    updates = tf.multiply(tf.cosh(nonzero_norms), updated_points) + tf.multiply(tf.sinh(nonzero_norms), normed_grads)
    return tf.scatter_update(vars, nonzero_indices, updates)


def tensor_exp_map2(vars, indices, tangent_grads):
    """
    Map vectors in the tangent space of the hyperboloid points back onto the hyperboloid
    :param hyperboloid_points: a tensor of points on the hyperboloid of shape (#examples, #dims)
    :param tangent_grads: a tensor of gradients on the tangent spaces of the hyperboloid_points of shape (#examples, #dims)
    :return:
    """
    # todo do we need to normalise the gradients?
    hyperboloid_points = tf.nn.embedding_lookup(vars, indices)
    norms = tf.sqrt(tf.maximum(minkowski_tensor_dot(tangent_grads, tangent_grads), 0))
    normed_grads = tf.divide(tangent_grads, norms)  # norms can be zero, so may contain nan and inf values
    values_to_replace = tf.logical_or(tf.is_nan(normed_grads), tf.is_inf(normed_grads))
    # we can replace with any finite value because sinh(0) = 0, so they cancel out
    safe_grads = tf.where(values_to_replace, tf.ones_like(normed_grads), normed_grads)
    updates = tf.multiply(tf.cosh(norms), hyperboloid_points) + tf.multiply(tf.sinh(norms), safe_grads)
    return tf.scatter_update(vars, indices, updates)


def tensor_exp_map(vars, indices, tangent_grads):
    """
    Map vectors in the tangent space of the hyperboloid points back onto the hyperboloid
    :param hyperboloid_points: a tensor of points on the hyperboloid of shape (#examples, #dims)
    :param tangent_grads: a tensor of gradients on the tangent spaces of the hyperboloid_points of shape (#examples, #dims)
    :return:
    """
    # todo do we need to normalise the gradients?
    hyperboloid_points = tf.nn.embedding_lookup(vars, indices)
    norms = tf.sqrt(tf.maximum(minkowski_tensor_dot(tangent_grads, tangent_grads), 0))
    normed_grads = tf.divide(tangent_grads, norms)  # norms can be zero, so may contain nan and inf values
    updates = tf.multiply(tf.cosh(norms), hyperboloid_points) + tf.multiply(tf.sinh(norms),
                                                                            normed_grads)  # norms can also be large sending sinh / cosh -> inf
    values_to_replace = tf.logical_or(tf.is_nan(updates), tf.is_inf(updates))
    safe_updates = tf.where(values_to_replace, hyperboloid_points, updates)
    return tf.scatter_update(vars, indices, safe_updates)


def transform_grads(grad):
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


def rsgd(grads, var, lr=1, max_norm=1):
    """
    Perform the Riemannian gradient descent operation by
    1/ Transforming gradients using the Minkowski metric tensor
    2/ Projecting onto the tangent space
    3/ Applying the exponential map
    :param grads:
    :param var:
    :param lr:
    :return:
    """
    grad, name = grads
    clipped_grads = tf.clip_by_norm(grad.values, max_norm, axes=1)
    # clipped_grads = grad.values
    minkowski_grads = transform_grads(clipped_grads)
    # minkowski_grads = transform_grads(grad.values)
    vecs = tf.nn.embedding_lookup(var, grad.indices)
    tangent_grads = project_tensors_onto_tangent_space(vecs, minkowski_grads)
    return tensor_exp_map(var, grad.indices, lr * tangent_grads)


def to_hyperboloid_points(poincare_pts):
    """
    Post: result.shape[1] == poincare_pts.shape[1] + 1
    """
    norm_sqd = (poincare_pts ** 2).sum(axis=1)
    # print('norm squared: ', norm_sqd)
    N = poincare_pts.shape[1]
    result = np.zeros((poincare_pts.shape[0], N + 1), dtype=np.float64)
    result[:, 1:] = (2. / (1. - norm_sqd))[:, np.newaxis] * poincare_pts
    result[:, 0] = (1. + norm_sqd) / (1. - norm_sqd)
    return result


def params_to_hyperboloid_points(vocab_size, embedding_size, init_width):
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


def to_poincare_ball_point(hyperboloid_pt):
    """
    project from hyperboloid to poincare ball
    :param hyperboloid_pt:
    :return:
    """
    N = len(hyperboloid_pt)
    return hyperboloid_pt[1:N] / (hyperboloid_pt[0])


def to_poincare_ball_points(hyperboloid_pts):
    """
    project from hyperboloid to poincare ball
    :param hyperboloid_pts:
    :return:
    """
    N = hyperboloid_pts.shape[1]
    return np.divide(hyperboloid_pts[:, 1:N], hyperboloid_pts[:, 0][:, None] + 1)


def get_logits(example, label, sample, true_b, sample_b):
    """
    produce logits to be used by the loss function
    :param example:
    :param label:
    :param sample:
    :param true_b:
    :param sample_b:
    :return:
    """
    true_logits = minkowski_dist(example, label) + true_b
    sampled_logits = pairwise_minkowski_dist(example, sample) + sample_b
    return true_logits, sampled_logits


def nce_loss(true_logits, sampled_logits):
    """
    The noise contrastive estimation loss function
    :param true_logits:
    :param sampled_logits:
    :return:
    """
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / 2
    return nce_loss_tensor


def nickel_initialisation(vocab_size, embedding_size, init_width=0.001):
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


def project_onto_manifold(tensor):
    """
    project a tensor back onto the hyperboloid by fixing the first coordinate to sqrt(|x[1:]|^2 + 1)
    :param tensor: a tensor of shape (examples, dimensions) where dimensions > 2
    :return:
    """
    kept_values = tensor[:, 1:]
    norm_square = tf.square(kept_values)
    new_vals = tf.expand_dims(tf.sqrt(tf.reduce_sum(norm_square, axis=1) + 1), axis=1)
    return tf.concat([new_vals, kept_values], axis=1)


def test_project_onto_manifold():
    init_value = tf.Variable([[1., 1., 1.], [2., -1., 2.], [3., 2., 3.], [4., 0., 4.]])
    retval = np.array([[-1.], [-1.], [-1.], [-1.]])
    tensor = project_onto_manifold(init_value)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    np.testing.assert_almost_equal(retval, sess.run(minkowski_tensor_dot(tensor, tensor)), decimal=5)


def test_to_poincare_ball_point():
    hyperboloid_point = np.array([1., 0., 0.])
    poincare_point = to_poincare_ball_point(hyperboloid_point)
    # choose random 3d poincare point
    random_poincare = 0.5 * np.random.uniform(size=(2, 3))  # 0.5 to keep points inside the 3-ball
    random_hyperboloid = to_hyperboloid_points(random_poincare)
    print('poincare:', random_poincare)
    print('hyperboloid:', random_hyperboloid)
    norms = minkowski_numpy_dot(random_hyperboloid, random_hyperboloid)
    retval = np.array([-1., -1.])
    assert np.array_equal(np.around(norms, 5), retval)
    print('norms: ', norms)
    print('back to poincare:', to_poincare_ball_points(random_hyperboloid))
    assert np.array_equal(poincare_point, np.array([0., 0.]))
    assert np.array_equal(np.around(random_poincare, 5), np.around(to_poincare_ball_points(random_hyperboloid), 5))


def test_to_poincare_ball_points():
    poincare_points = 0.5 * np.random.uniform(size=(2, 3))  # 0.5 to keep points inside the 3-ball
    hyperboloid_points = to_hyperboloid_points(poincare_points)
    norms = minkowski_numpy_dot(hyperboloid_points, hyperboloid_points)
    # print('norms: ', norms, type(norms), norms.shape)
    retval = np.array([-1., -1.])
    # print('retval: ', retval, type(retval), retval.shape)
    assert np.array_equal(np.around(norms, 5), retval)


def test_project_onto_tangent_space():
    """
    Remember that we negate the first co-ordinate, so in 2D the hyperboloid appears rotated 90 degrees clockwise from how it is normally drawn
    :return:
    """
    y = tf.constant([1., 0])  # this the minima of the hyperboloid
    z = tf.constant([1., 0])
    w = tf.constant([0., 1.])
    u = tf.constant([0., -1.])
    v = tf.constant([1., 1.])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # here the tangent space is x=1
    assert np.array_equal(sess.run(project_onto_tangent_space(y, z)), np.array([0., 0.]))
    assert np.array_equal(sess.run(project_onto_tangent_space(y, w)), np.array([0., 1.]))
    assert np.array_equal(sess.run(project_onto_tangent_space(y, u)), np.array([0., -1.]))
    assert np.array_equal(sess.run(project_onto_tangent_space(y, v)), np.array([0., 1.]))
    # consecutive projections don't change values
    s = tf.constant([math.sqrt(5), 2])  # point on hyperboloid
    t = tf.constant([6.2, 3.2])  # random grad
    temp1 = sess.run(project_onto_tangent_space(s, t))
    temp2 = sess.run(project_onto_tangent_space(s, temp1))
    assert np.array_equal(temp1, temp2)


def test_project_tensors_onto_tangent_space():
    u1 = tf.constant([[1., 0.], [1., 0.], [1., 0.], [1., 0.]], dtype=tf.float32)
    v1 = tf.constant([[1., 0.], [0., 1.], [0., -1.], [1., 1.]], dtype=tf.float32)
    retval1 = np.array([[0., 0.], [0., 1.], [0., -1.], [0., 1.]])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert np.array_equal(sess.run(project_tensors_onto_tangent_space(u1, v1)), retval1)
    # consecutive projections don't change values
    temp1 = sess.run(project_tensors_onto_tangent_space(u1, v1))
    temp2 = sess.run(project_tensors_onto_tangent_space(u1, temp1))
    assert np.array_equal(temp1, temp2)


def test_exp_map():
    """
    check that the exp_map takes vectors in the tangent space to the manifold
    :return:
    """
    p1 = tf.constant([1., 0.])  # this the minima of the hyperboloid
    g1 = tf.constant([0., 1.])
    g2 = tf.constant([0., -1.])
    g3 = tf.constant([0., 2.])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # here the tangent space is x=1
    em1 = exp_map(p1, g1)
    em2 = exp_map(p1, g2)
    em3 = exp_map(p1, g3)
    # check that the points are on the hyperboloid
    assert round(sess.run(minkowski_vector_dot(em1, em1)), 4) == -1.
    assert round(sess.run(minkowski_vector_dot(em2, em2)), 4) == -1.
    assert round(sess.run(minkowski_vector_dot(em3, em3)), 4) == -1.
    em1 = sess.run(exp_map(p1, g1))
    em2 = sess.run(exp_map(p1, g2))
    em3 = sess.run(exp_map(p1, g3))
    assert em1[0] == em2[0]
    assert em1[1] == -em2[1]
    assert em3[0] > em1[0]
    assert em3[1] > em1[1]


def test_minkowski_vector_dot():
    u1 = tf.constant([1., 0])
    v1 = tf.constant([1., 0])
    v2 = tf.constant([10., 0])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert sess.run(minkowski_vector_dot(u1, v1)) == -1
    assert sess.run(minkowski_vector_dot(u1, v2)) == -10


def test_minkowski_tensor_dot():
    u1 = tf.constant([[1., 0., 1.], [1., 1., 1.]], dtype=tf.float32)
    v1 = tf.constant([[1., 1., 1.], [0., 0., 1.]], dtype=tf.float32)
    v2 = tf.constant([[2., 1., 1.], [2., 0., 1.]], dtype=tf.float32)
    u4 = tf.constant([[1.00000012e+00, 3.24688008e-04, 3.21774220e-04]])
    v4 = tf.constant([[1.00000000e+00, 3.61118466e-04, 1.88534090e-04]])  # this one should have an invalid norm
    retval1 = np.array([[0.], [1.]])
    retval2 = np.array([[-1.], [-1.]])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    u4_norm = sess.run(minkowski_tensor_dot(u4, u4))
    v4_norm = sess.run(minkowski_tensor_dot(v4, v4))
    print('u4 norm: ', u4_norm)
    print('v4 norm: ', v4_norm)
    dot4 = sess.run(minkowski_tensor_dot(u4, v4))
    print('dot4: ', dot4)
    assert np.array_equal(sess.run(minkowski_tensor_dot(u1, v1)), retval1)
    assert np.array_equal(sess.run(minkowski_tensor_dot(u1, v2)), retval2)
    assert u4_norm == -1.
    assert v4_norm == -0.99999982


def test_pairwise_minkowski_dot():
    # D_ij = x_i * y_j
    x = np.array([[1, 0], [0, 1]])
    y = np.array([[3, 4], [5, 6]])
    u = tf.Variable(x, dtype=tf.float32)
    v = tf.Variable(y, dtype=tf.float32)
    retval = np.array([[-3., -5.], [4., 6.]])
    z = pairwise_minkowski_dot(u, v)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert np.array_equal(sess.run(z), retval)


def test_minkowski_dist():
    u = tf.constant([[1., 0], [1., 0.]])
    v = tf.constant([[1., 0], [10., 0.]])
    # x = tf.constant([10., 0])
    N = 100
    poincare_pts1 = np.divide(np.random.rand(N, 2), np.sqrt(2))
    poincare_pts2 = np.divide(np.random.rand(N, 2), np.sqrt(2))
    hyp_points1 = tf.Variable(to_hyperboloid_points(poincare_pts1))
    hyp_points2 = tf.Variable(to_hyperboloid_points(poincare_pts2))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    dist = sess.run(minkowski_dist(u, v))
    assert dist[0] == 0
    assert dist[1] != 0
    # generate some points on the hyperboloid
    distances = sess.run(minkowski_dist(hyp_points1, hyp_points2))
    # print(distances)
    assert np.sum(distances <= 0) == 0


def test_pairwise_minkowski_dist():
    u = tf.constant([[1., 0], [1., 0.]])
    v = tf.constant([[1., 0], [10., 0.]])
    u2 = tf.constant([[1.00000012e+00, -4.40269214e-05, -3.25769943e-05]])
    v2 = tf.constant([[1.00000000e+00, 1.49380328e-04, -5.60168264e-05]])
    u3 = tf.constant([[1.00000072e+00, 8.21528956e-04, 9.05358582e-04]])
    v3 = tf.constant([[2.08181834, 0.85871041, 1.61139297]])
    u4 = tf.constant([[1.00000012e+00, 3.24688008e-04, 3.21774220e-04]])
    v4 = tf.constant([[1.00000000e+00, 3.61118466e-04,
                       1.88534090e-04]])  # this is slightly off the manifold and causes nan distances
    # x = tf.constant([10., 0])
    N = 100
    poincare_pts1 = np.divide(np.random.rand(N, 2), np.sqrt(2))
    poincare_pts2 = np.divide(np.random.rand(N, 2), np.sqrt(2))
    hyp_points1 = tf.Variable(to_hyperboloid_points(poincare_pts1))
    hyp_points2 = tf.Variable(to_hyperboloid_points(poincare_pts2))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    dist = sess.run(pairwise_minkowski_dist(u, v))
    dist2 = sess.run(pairwise_minkowski_dist(u2, v2))
    print('dist2: ', dist2)
    dist3 = sess.run(pairwise_minkowski_dist(u3, v3))
    print('dist3: ', dist3)
    dist4 = sess.run(pairwise_minkowski_dist(u4, v4))
    print('dist4: ', dist4)

    assert dist.shape == (2, 2)
    assert dist[0, 0] == 0.
    assert dist[1, 0] == 0.
    assert dist[1, 1] != 0.
    assert ~np.isnan(dist2)
    assert ~np.isnan(dist3)
    assert dist4 == 0
    # generate some points on the hyperboloid
    distances = sess.run(pairwise_minkowski_dist(hyp_points1, hyp_points2))
    # print(distances)
    assert np.sum(distances <= 0) == 0
    assert distances.shape == (N, N)
    print('max dist: ', np.max(distances[:]))
    print('min dist: ', np.min(distances[:]))


def test_gradient_transform_single_vector():
    grads = tf.Variable(np.array([1, 2]), dtype=tf.int32)
    x = np.eye(grads.shape[0])
    true_val = np.array([-1, 2])
    x[0, 0] = -1
    T = tf.constant(x, dtype=tf.int32)
    U = tf.tensordot(grads, T, 1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    retval = sess.run(U)
    assert (np.array_equal(retval, true_val))


def test_gradient_transform_matrix():
    """
    grads come in a matrix of shape (n_vars, embedding_dim)
    :return:
    """
    grads = tf.Variable(np.array([[1., 2., 0.2], [3., 4., 0.5], [4., 2., 0.6]]))
    true_val = np.array([[-1., 2., 0.2], [-3., 4., 0.5], [-4., 2., 0.6]])
    x = np.eye(grads.shape[1])
    x[0, 0] = -1.
    T = tf.constant(x, dtype=tf.float64)
    U = tf.matmul(grads, T)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    retval = sess.run(U)
    assert (np.array_equal(retval, true_val))


def test_transform_grads():
    g1 = tf.constant([[1., 1.], [2., -1.], [3., 2.], [4., 0.]])
    g2 = tf.constant([1., 2., 3.])
    retval1 = np.array([[-1., 1.], [-2., -1.], [-3., 2.], [-4., 0.]])
    retval2 = np.array([[-1., 2., 3.]])
    transformed_grads1 = transform_grads(g1)
    transformed_grads2 = transform_grads(g2)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert np.array_equal(sess.run(transformed_grads1), retval1)
    assert np.array_equal(sess.run(transformed_grads2), retval2)


def test_tensor_exp_map():
    """
    check that the exp_map takes vectors in the tangent space to the manifold
    :return:
    """
    input_points = np.array([[1., 0.], [1., 0.], [4., 5.], [1., 0.], [1., 0.], [1., 0.]])
    p1 = tf.Variable(input_points, dtype=tf.float32)  # this the minima of the hyperboloid
    indices = tf.constant([0, 1, 3, 4, 5])
    g1 = tf.constant([[0., 1.], [0., -1.], [0., 2.], [0., 0.], [1., 0.]])
    retval1 = np.array([[-1.], [-1.], [-1.], [-1.], [-1.]])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # here the tangent space is x=1
    new_vars = tensor_exp_map(p1, indices, g1)
    em1 = tf.nn.embedding_lookup(new_vars, indices)
    # check that the points are on the hyperboloid
    norms = sess.run(minkowski_tensor_dot(em1, em1))
    assert np.array_equal(np.around(norms, 3), retval1)
    em1 = sess.run(em1)
    new_vars = sess.run(new_vars)
    np_new_vars = np.array(new_vars)
    assert np.array_equal(np_new_vars[2, :], input_points[2, :])
    assert np.array_equal(np_new_vars[4, :], input_points[4, :]), 'point with |g|<= 0 was updated'
    assert np.array_equal(np_new_vars[5, :], input_points[5, :]), 'point with |g|<= 0 was updated'
    assert em1[0, 0] == em1[1, 0]
    assert em1[0, 1] == -em1[1, 1]
    assert em1[2, 0] > em1[0, 0]
    assert em1[2, 1] > em1[0, 1]


def test_rsgd():
    from collections import namedtuple
    grad = namedtuple('grad', 'values indices')
    input_points = np.array([[1., 0.], [1., 0.], [4., 5.], [1., 0.], [1., 0.]])
    p1 = tf.Variable(input_points, dtype=tf.float32)  # this the minima of the hyperboloid
    indices = tf.constant([0, 1, 3, 4])
    g1 = tf.constant([[0., 1.], [0., -1.], [0., 2.], [0., 0.]])
    grad.values = g1
    grad.indices = indices
    grads = (grad, 'test_grads')
    retval1 = np.array([[-1.], [-1.], [-1.], [-1.]])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # here the tangent space is x=1
    p2 = rsgd(grads, p1, lr=1.)
    updated_points = tf.nn.embedding_lookup(p2, indices)
    # check that the points are on the hyperboloid
    norms = sess.run(minkowski_tensor_dot(updated_points, updated_points))
    assert np.array_equal(np.around(norms, 3), retval1)
    new_vars = sess.run(p2)
    np_new_vars = np.array(new_vars)
    assert np.array_equal(np_new_vars[2, :], input_points[2, :])
    assert np.array_equal(np_new_vars[4, :], input_points[4, :])


def test_to_hyperboloid_points(N=100):
    poincare_pts = np.divide(np.random.rand(N, 2), np.sqrt(2))  # sample a load of points in the Poincare disk
    hyp_points = tf.Variable(to_hyperboloid_points(poincare_pts))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert np.array_equal(np.around(sess.run(minkowski_tensor_dot(hyp_points, hyp_points)), 3), np.array(N * [[-1.]]))


def test_nickel_initialisation():
    N = 100
    embedding_size = 3
    hyp_points = tf.Variable(nickel_initialisation(N, embedding_size))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    np.testing.assert_array_almost_equal(sess.run(minkowski_tensor_dot(hyp_points, hyp_points)), np.array(N * [[-1.]]))


def test_grads_vectors():
    """
    tests the gradients of the pairwise and elementwise distance functions with 1 sample, 1 example and 1 label
    :return:
    """
    embedding_size = 2
    vocab_size = 4
    emb = tf.Variable(params_to_hyperboloid_points(vocab_size, embedding_size, 0.1), dtype=tf.float32)
    sm_w_t = tf.Variable(params_to_hyperboloid_points(vocab_size, embedding_size, 0.1), dtype=tf.float32)
    # sm_w_t = tf.Variable(tf.zeros([vocab_size, embedding_size]))
    sm_b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

    examples = tf.Variable([1, 2])
    labels = tf.Variable([2, 2])
    sampled_ids = tf.Variable([1, 3])

    example_emb = tf.nn.embedding_lookup(emb, examples)
    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    print('emb shape: ', example_emb.shape)
    print('sample w shape: ', sampled_w.shape)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
    true_logits, sampled_logits = get_logits(example_emb, true_w, sampled_w, true_b, sampled_b)
    loss = nce_loss(true_logits, sampled_logits)
    opt = tf.train.GradientDescentOptimizer(0.1)
    emb_grad = opt.compute_gradients(loss, [emb])
    sm_w_t_grad = opt.compute_gradients(loss, [sm_w_t])
    grads = emb_grad + sm_w_t_grad
    apply_grad = opt.apply_gradients(grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('original vectors: ', sess.run([example_emb, true_w]))
        print('emb grads are: ', sess.run(emb_grad))
        print('sm_w_t grads are: ', sess.run(sm_w_t_grad))
        sess.run(apply_grad)
        print('updated vectors: ', sess.run([example_emb, true_w]))


def test_moving_along_hyperboloid():
    """
    test multiple series of updates
    :return:
    """
    from collections import namedtuple
    learning_rate = 0.5
    max_grad_norm = 1.
    grad = namedtuple('grad', 'values indices')
    input_points = np.array([[1., 0.], [1., 0.], [4., 5.], [1., 0.], [1., 0.]], dtype=np.float32)
    p1 = tf.Variable(input_points, dtype=tf.float32)  # this the minima of the hyperboloid
    indices = tf.constant([0, 1, 3, 4])
    g1 = tf.constant([[0., 1.], [0., -1.], [1., 2.], [0., 0.]])
    grad.values = g1
    grad.indices = indices
    grads = (grad, 'test_grads')
    retval1 = np.array([[-1.], [-1.], [-1.], [-1.]])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # here the tangent space is x=1
    pvals = []
    p1 = rsgd(grads, p1, lr=learning_rate, max_norm=max_grad_norm)
    for counter in range(20):
        # print(sess.run(p1))
        updated_points = tf.nn.embedding_lookup(p1, indices)
        # check that the points are on the hyperboloid
        norms = minkowski_tensor_dot(updated_points, updated_points)
        print('counter: ', counter)
        # print('norms: ', norms)
        points_vals, norms_vals = sess.run([p1, norms])
        print('new points: ', points_vals)
        print('norms: ', norms_vals)
        # p1_val, norms_val = sess.run([p1, norms])
        # print(p1_val, norms_val)
        # assert np.array_equal(np.around(norms_val, 3), retval1)
        # new_vars = sess.run(p2)
        # np_new_vars = np.array(new_vars)
        assert np.array_equal(points_vals[2, :], input_points[2, :])
        assert np.array_equal(points_vals[4, :], input_points[4, :])


if __name__ == '__main__':
    test_gradient_transform_matrix()
