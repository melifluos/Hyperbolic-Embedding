"""
Tests that the hyperbolic cartesian gradients are being calculated properly
"""
import sys
import os

sys.path.append(os.path.join('..', 'src', 'python'))

import utils
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from run_detectors import run_all_datasets
import numpy as np
import tensorflow as tf
import math


def grad_d_x(x, y):
    """
    dD(x,y)/dx where D(x,y) gives the hyperbolic distance between two vectors in the Poincare ball that satisfy ||x||^2<1
    :param x: vector
    :param y: vector
    :return: gradient
    """
    alpha = 1 - np.square(np.linalg.norm(x, axis=0))
    beta = 1 - np.square(np.linalg.norm(y, axis=0))
    norm_sq = np.square(np.linalg.norm(x - y, axis=0))
    gamma = 1.0 + 2.0 * norm_sq / (alpha * beta)
    fac1 = 4.0 / (beta * np.sqrt(np.square(gamma) - 1))
    y_norm_sq = np.square(np.linalg.norm(y, axis=0))
    fac2 = y_norm_sq - 2 * np.dot(x, y) + 1
    # the first term in the brackets
    fac3 = (fac2 * x) / np.square(alpha)
    grad = fac1 * (fac3 - y / alpha)
    return grad


def np_distance(x, y):
    """
    The distance between two vectors
    :param x: shape (1, ndims)
    :param y: shape (1,ndims)
    :return: a scalar hyperbolic distance
    """
    # assert np.round(norm_square, 5) == 0.16
    denom1 = 1 - np.square(np.linalg.norm(x))
    print denom1
    # assert np.round(denom1, 5) == 0.91
    if len(y.shape) > 1:
        norm_square = np.square(np.linalg.norm(x - y, axis=1))
        denom2 = 1 - np.square(np.linalg.norm(y, axis=1))
    else:
        norm_square = np.square(np.linalg.norm(x - y, axis=0))
        denom2 = 1 - np.square(np.linalg.norm(y, axis=0))
    print norm_square
    print denom2
    # assert np.round(denom2, 5) == 0.75
    arg = 1.0 + 2.0 * norm_square / (denom1 * denom2)
    print arg
    return np.float32(np.arccosh(arg))


def tf_distance(x, y):
    """
    The distance between two vectors
    :param x: shape (1, ndims)
    :param y: shape (1,ndims)
    :return: a scalar hyperbolic distance
    """
    norm_square = tf.square(tf.norm(x - y, axis=0))
    print norm_square
    denom1 = 1 - tf.square(tf.norm(x, axis=0))
    print denom1
    denom2 = 1 - tf.square(tf.norm(y, axis=0))
    print denom2
    arg = 1 + 2 * norm_square / (denom1 * denom2)
    print arg
    return tf.acosh(arg)


def elementwise_distance(examples, labels):
    """
    creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
    :param examples: first set of vectors of shape (ndata1, ndim)
    :param labels: second set of vectors of shape (ndata2, ndim)
    :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
    """
    xnorm_sq = tf.square(tf.norm(examples, axis=1))
    ynorm_sq = tf.square(tf.norm(labels, axis=1))
    euclidean_dist_sq = tf.square(tf.norm(examples - labels, axis=1))
    denom = tf.multiply(1 - xnorm_sq, 1 - ynorm_sq)
    hyp_dist = tf.acosh(1 + 2 * tf.divide(euclidean_dist_sq, denom))
    return hyp_dist


def pairwise_distance(examples, samples):
    """
    creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
    :param examples: first set of vectors of shape (ndata1, ndim)
    :param samples: second set of vectors of shape (ndata2, ndim)
    :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
    """
    xnorm_sq = tf.square(tf.norm(examples, axis=1))
    ynorm_sq = tf.square(tf.norm(samples, axis=1))
    # use the multiplied out version of the l2 norm to simplify broadcasting ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy.T
    euclidean_dist_sq = xnorm_sq[:, None] + ynorm_sq[None, :] - 2 * tf.matmul(examples, samples, transpose_b=True)
    denom = (1 - xnorm_sq[:, None]) * (1 - ynorm_sq[None, :])
    hyp_dist = tf.acosh(1 + 2 * tf.divide(euclidean_dist_sq, denom))
    return hyp_dist


def get_logits(example, label, sample, true_b, sample_b):
    true_logits = tf_distance(example, label) + true_b
    sampled_logits = tf_distance(example, sample) + sample_b
    return true_logits, sampled_logits


def get_tensor_logits(examples, labels, samples, true_b, sample_b):
    true_logits = elementwise_distance(examples, labels) + true_b
    sampled_logits = pairwise_distance(examples, samples) + sample_b
    return true_logits, sampled_logits


def nce_loss(true_logits, sampled_logits, batch_size=1):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # NCE-
    #  is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / batch_size
    return nce_loss_tensor


def test_logits():
    example = tf.Variable([0.1, 0.0])
    label = tf.Variable([.0, .3])
    sample = tf.Variable([.3, .2])
    true_b = 0
    sample_b = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(get_logits(example, label, sample, true_b, sample_b))


def test_loss():
    example = tf.Variable([0.1, 0.0])
    label = tf.Variable([.0, .3])
    sample = tf.Variable([.3, .2])
    true_b = 0
    sample_b = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        true_logits, sampled_logits = sess.run(get_logits(example, label, sample, true_b, sample_b))
        loss = sess.run(nce_loss(true_logits, sampled_logits))
        print loss


def test_grads():
    """
    tests the gradients of the simple vector distance function
    :return:
    """
    example = tf.Variable([0.1, 0.3])
    label = tf.Variable([.0, .3])
    sample = tf.Variable([.2, .1])
    true_b = 0
    sample_b = 0
    true_logits, sampled_logits = get_logits(example, label, sample, true_b, sample_b)
    loss = nce_loss(true_logits, sampled_logits)
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads = opt.compute_gradients(loss, [example, label])
    apply_grad = opt.apply_gradients(grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run([example, label])
        print 'grads are: ', sess.run(grads)
        sess.run(apply_grad)
        print sess.run([example, label])


def test_grad_clipping():
    """
    tests the gradients of the simple vector distance function
    :return:
    """
    # this configuration will have an infinite gradient becuase the example and label vecs are equal
    example = tf.Variable([0.0001, 0.299])
    label = tf.Variable([.0, .3])
    sample = tf.Variable([.2, .1])
    true_b = 0
    sample_b = 0
    true_logits, sampled_logits = get_logits(example, label, sample, true_b, sample_b)
    loss = nce_loss(true_logits, sampled_logits)
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads = opt.compute_gradients(loss, [example, label])
    clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]

    apply_grad = opt.apply_gradients(clipped_grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'input tensors: ', sess.run([example, label])
        print 'grads are: ', sess.run([clipped_grads])
        sess.run(apply_grad)
        print 'final tensors: ', sess.run([example, label])


def remove_nan(tensor, epsilon=0.001):
    """
    replace nans in gradients caused by comparing two indentical vectors with a small gradient
    :param tensor:
    :param epsilon:
    :return:
    """
    return tf.where(tf.is_nan(tensor), tf.ones_like(tensor) * epsilon, tensor)


def remove_index_slices_nan(grad, epsilon=0.001):
    """
    replace nans in gradients caused by comparing two indentical vectors with a small gradient
    :param grad: a tf.IndexedSlicesValue
    :param epsilon: the value to set nan values to
    :return: the tensor with nans replaced by epsilon
    """
    g = tf.where(tf.is_nan(grad.values), tf.ones_like(grad.values) * epsilon, grad.values)
    safe_grad = tf.IndexedSlices(g, grad.indices, grad.dense_shape)
    return safe_grad


def test_nan_removal():
    """
    tests the gradients of the simple vector distance function
    :return:
    """
    # this configuration will have an infinite gradient becuase the example and label vecs are equal
    example = tf.Variable([0., 0.3])
    label = tf.Variable([.0, .3])
    sample = tf.Variable([.2, .1])
    true_b = 0
    sample_b = 0
    true_logits, sampled_logits = get_logits(example, label, sample, true_b, sample_b)
    loss = nce_loss(true_logits, sampled_logits)
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads = opt.compute_gradients(loss, [example, label])
    clipped_grads = [(remove_nan(grad), var) for grad, var in grads]

    apply_grad = opt.apply_gradients(clipped_grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'input tensors: ', sess.run([example, label])
        print 'grads are: ', sess.run([clipped_grads])
        sess.run(apply_grad)
        final_tensors = sess.run([example, label])
        print final_tensors
        assert ~np.isnan(np.sum(final_tensors)), "nan value found in final tensors"


def test_remove_nan_from_index_slices():
    """
    tests the gradients of the simple vector distance function
    :return:
    """
    # this configuration will have an infinite gradient becuase the example and label vecs are equal
    examples = tf.Variable([[0., 0.3], [1, 2]])
    labels = tf.Variable([[.0, .3], [2, 3]])
    example = tf.nn.embedding_lookup(examples, 0)
    # Weights for labels: [batch_size, emb_dim]
    label = tf.nn.embedding_lookup(labels, 0)

    sample = tf.Variable([.2, .1])
    true_b = 0
    sample_b = 0
    true_logits, sampled_logits = get_logits(example, label, sample, true_b, sample_b)
    loss = nce_loss(true_logits, sampled_logits)
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads = opt.compute_gradients(loss, [examples, labels])
    clipped_grads = [(remove_index_slices_nan(grad), var) for grad, var in grads]

    apply_grad = opt.apply_gradients(clipped_grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'input tensors: ', sess.run([examples, labels])
        print 'grads are: ', sess.run(grads)
        print 'clipped grads are: ', sess.run(clipped_grads)
        sess.run(apply_grad)
        final_tensors = sess.run([examples, labels])
        print 'final tensors: ', final_tensors
        assert ~np.isnan(np.sum(final_tensors)), "nan value found in final tensors"


def test_grads_vectors():
    """
    tests the gradients of the pairwise and elementwise distance functions with 1 sample, 1 example and 1 label
    :return:
    """
    embedding_size = 2
    vocab_size = 4
    emb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1))
    sm_w_t = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1))
    # sm_w_t = tf.Variable(tf.zeros([vocab_size, embedding_size]))
    sm_b = tf.Variable(tf.zeros([vocab_size]))

    examples = tf.Variable([1])
    labels = tf.Variable([2])
    sampled_ids = tf.Variable([3])

    example_emb = tf.nn.embedding_lookup(emb, examples)
    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
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
        print 'original vectors: ', sess.run([example_emb, true_w])
        print 'emb grads are: ', sess.run(emb_grad)
        print 'sm_w_t grads are: ', sess.run(sm_w_t_grad)
        sess.run(apply_grad)
        print 'updated vectors: ', sess.run([example_emb, true_w])


def test_clip_tensor_norms(epsilon=0.00001):
    """
    not used as clip_by_norm performs this task
    :param epsilon:
    :return:
    """
    emb = tf.Variable([[1.9, 0.299], [2.1, 3.4]])
    norms = tf.norm(emb, axis=1)
    comparison = tf.greater(norms, tf.constant(1.0, dtype=tf.float32))
    norm = tf.nn.l2_normalize(emb, dim=1) - epsilon
    conditional_assignment_op = tf.assign(emb, tf.where(comparison, norm, emb))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'inital tensor: ', sess.run(emb)
        sess.run(conditional_assignment_op)
        print 'final tensor: ', sess.run(emb)


def test_clip_indexed_slices_norms(epsilon=0.00001):
    """
    not used as clip_by_norm performs this task
    :param epsilon:
    :return:
    """
    emb = tf.Variable([[1.9, 0.299], [2.1, 3.4], [0.1, 0.3]])
    examples = tf.constant([1, 2])
    emb_slice = tf.nn.embedding_lookup(emb, examples)
    norms = tf.norm(emb_slice, axis=1)
    comparison = tf.greater(norms, tf.constant(1.0, dtype=tf.float32))
    norm = tf.nn.l2_normalize(emb_slice, dim=1) - epsilon
    normed_slice = tf.where(comparison, norm, emb_slice)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'inital tensor: ', sess.run(emb_slice)
        # sess.run(conditional_assignment_op)
        print 'final tensor: ', sess.run(normed_slice)


class TestEmbClipping:
    """
    test that the embedding clipping is being called
    :return:
    """

    def __init__(self):
        self.build_graph()

    def build_graph(self):
        """
        construct a forward pass
        :return:
        """
        embedding_size = 2
        vocab_size = 6
        self.emb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -10, 10))
        self.sm_w_t = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1))
        # sm_w_t = tf.Variable(tf.zeros([vocab_size, embedding_size]))
        self.sm_b = tf.Variable(tf.zeros([vocab_size]))

        examples = tf.constant([0, 1])
        labels = tf.constant([2, 3])
        sampled_ids = tf.constant([4, 5])

        example_emb = tf.nn.embedding_lookup(self.emb, examples)
        # make sure tensors are normalised
        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(self.sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(self.sm_b, labels)
        sampled_w = tf.nn.embedding_lookup(self.sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(self.sm_b, sampled_ids)
        true_logits, sampled_logits = get_tensor_logits(example_emb, true_w, sampled_w, true_b, sampled_b)
        loss = nce_loss(true_logits, sampled_logits)
        self._loss = loss
        self.optimize(loss)
        self.clip_emb_tensor_norms()
        self.clip_sm_tensor_norms()

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

    def optimize(self, loss):
        """
        optimize a loss function
        :param loss:
        :return:
        """
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        # emb = self.clip_tensor_norms(self.emb)
        # sm_w_t_grad =
        # emb = self.clip_tensor_norms()
        emb_grad = optimizer.compute_gradients(loss, [self.emb])
        # emb_grad = self.modify_grads(temp, emb)
        sm_w_t_grad = optimizer.compute_gradients(loss, [self.sm_w_t])
        grads = emb_grad + sm_w_t_grad
        self._train = optimizer.apply_gradients(grads)

    def clip_emb_tensor_norms(self, epsilon=0.00001):
        """
        not used as clip_by_norm performs this task
        :param epsilon:
        :return:
        """
        norms = tf.norm(self.emb, axis=1)
        comparison = tf.greater(norms, tf.constant(1.0, dtype=tf.float32))
        norm = tf.nn.l2_normalize(self.emb, dim=1) - epsilon
        # tf.assign(emb, tf.where(comparison, norm, emb))
        # self.emb = emb
        self._emb_clipper = self.emb.assign(tf.where(comparison, norm, self.emb))
        # conditional_assignment_op = emb.assign(tf.where(comparison, norm, emb))
        # return conditional_assignment_op

    def clip_sm_tensor_norms(self, epsilon=0.00001):
        """
        not used as clip_by_norm performs this task
        :param epsilon:
        :return:
        """
        norms = tf.norm(self.sm_w_t, axis=1)
        comparison = tf.greater(norms, tf.constant(1.0, dtype=tf.float32))
        norm = tf.nn.l2_normalize(self.sm_w_t, dim=1) - epsilon
        # tf.assign(emb, tf.where(comparison, norm, emb))
        # self.emb = emb
        self._sm_clipper = self.emb.assign(tf.where(comparison, norm, self.sm_w_t))

    def clip_tensor_norms(self, emb, epsilon=0.00001):
        """
        not used as clip_by_norm performs this task
        :param epsilon:
        :return:
        """
        norms = tf.norm(emb, axis=1)
        comparison = tf.greater(norms, tf.constant(1.0, dtype=tf.float32))
        norm = tf.nn.l2_normalize(emb, dim=1) - epsilon
        # tf.assign(emb, tf.where(comparison, norm, emb))
        # self.emb = emb
        self._clipper = self.emb.assign(tf.where(comparison, norm, emb))


def test_emb_clipping():
    with tf.Session() as sess:
        test = TestEmbClipping()
        sess.run(tf.global_variables_initializer())
        for epoch in range(3):
            emb, sm_w_t = sess.run([test.emb, test.sm_w_t])
            print emb
            print sm_w_t
            sess.run(test._clipper(test.emb))
            sess.run(test._clipper(test.sm_w_t))
            emb, sm_w_t = sess.run([test.emb, test.sm_w_t])
            print emb
            print sm_w_t
            # print temp
            # sess.run(test._train)
            sess.run(test._train)
            # tensors = sess.run([example_emb, true_w, sampled_w])
            # # assert ~np.isnan(np.sum(tensors)), "nan value found in tensors"
            # grads = sess.run([emb_grad, sm_w_t_grad])
            # emb_grads = grads[0][0][0]
            # sm_w_t_grads = grads[1][0][0]
            # assert ~np.isnan(np.sum(emb_grads.values)), \
            #     "nan value found in grads at epoch {} with {} tensors for emb grads {}".format(epoch, tensors,
            #                                                                                    emb_grads)
            # assert ~np.isnan(np.sum(sm_w_t_grads.values)), \
            #     "nan value found in grads at epoch {} with {} tensors for sm grads {}".format(epoch, tensors,
            #                                                                                   sm_w_t_grads)
            # sess.run(apply_grad)


def test_grads_tensors():
    """
    tests the gradients of the pairwise and elementwise distance functions with 1 sample, 1 example and 1 label
    :return:
    """
    embedding_size = 2
    vocab_size = 100
    emb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1))
    sm_w_t = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1))
    # sm_w_t = tf.Variable(tf.zeros([vocab_size, embedding_size]))
    sm_b = tf.Variable(tf.zeros([vocab_size]))

    examples = tf.Variable([1, 2])
    labels = tf.Variable([3, 4])
    sampled_ids = tf.Variable([5, 6, 7, 8, 9, 10, 11, 12])

    example_emb = tf.nn.embedding_lookup(emb, examples)
    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
    true_logits, sampled_logits = get_tensor_logits(example_emb, true_w, sampled_w, true_b, sampled_b)
    loss = nce_loss(true_logits, sampled_logits)
    opt = tf.train.GradientDescentOptimizer(0.1)
    emb_grad = opt.compute_gradients(loss, [emb])
    sm_w_t_grad = opt.compute_gradients(loss, [sm_w_t])
    grads = emb_grad + sm_w_t_grad
    apply_grad = opt.apply_gradients(grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            tensors = sess.run([example_emb, true_w, sampled_w])
            # assert ~np.isnan(np.sum(tensors)), "nan value found in tensors"
            grads = sess.run([emb_grad, sm_w_t_grad])
            emb_grads = grads[0][0][0]
            sm_w_t_grads = grads[1][0][0]
            assert ~np.isnan(np.sum(emb_grads.values)), \
                "nan value found in grads at epoch {} with {} tensors for emb grads {}".format(epoch, tensors,
                                                                                               emb_grads)
            assert ~np.isnan(np.sum(sm_w_t_grads.values)), \
                "nan value found in grads at epoch {} with {} tensors for sm grads {}".format(epoch, tensors,
                                                                                              sm_w_t_grads)
            sess.run(apply_grad)


def test_slice_update():
    """
    test updating just a slice of a tensor
    :return:
    """
    embedding_size = 2
    vocab_size = 100
    emb = tf.Variable(tf.zeros([3, 4]))
    indices = tf.Variable([0, 1])
    slice_assign = emb[indices, :].assign(tf.ones(4))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(emb)
        print sess.run(slice_assign)


if __name__ == '__main__':
    test_clip_indexed_slices_norms()
    # test_clip_tensor_norms()
    # test_emb_clipping()
    # test_grads_tensors()
    # x = np.array([0.0, 0.999])
    # y = np.array([.5, 0.0])
    # print grad_d_x(x, y)
