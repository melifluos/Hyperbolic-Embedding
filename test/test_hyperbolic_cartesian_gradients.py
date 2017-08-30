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


def forward(examples, labels, opts):
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
        self.sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size, opts.embedding_size]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        self.sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size, 1])

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(self.emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(self.sm_w_t, labels)
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
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(self.sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        # true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
        true_logits = self.elementwise_distance(example_emb, true_w) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        # sampled_logits = self.pairwise_distance(example_emb, sampled_w) + sampled_b_vec
    return true_logits, sampled_logits


def get_logits(example, label, sample, true_b, sample_b):
    true_logits = tf_distance(example, label) + true_b
    sampled_logits = tf_distance(example, sample) + sample_b
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
    tests the gradients of the pairwise and elementwise distance functions.
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


def test_NCE_theta():
    """
    testing the NCE estimation component in 1D angular co-ordinates without modifying the gradient
    :return:
    """
    example = 1
    theta1 = tf.Variable(math.pi / 3.0, name='theta1')
    theta2 = tf.Variable(math.pi / 6.0, name='theta2')
    theta3 = tf.Variable(math.pi / 9.0, name='theta3')
    theta4 = tf.Variable(math.pi / 12.0, name='theta4')
    # radius1 = tf.Variable(1.0)
    # radius2 = tf.Variable(1.0)
    # radius3 = tf.Variable(1.0)
    # radius4 = tf.Variable(1.0)
    b = tf.Variable(1.0, name='b')

    def tf_logits(theta1, theta2):
        return tf.cos(theta1 - theta2)

    def logit(theta1, theta2):
        return np.cos(theta1 - theta2)

    def grad_u(theta1, theta2):
        return -1.0 * np.sin(theta1 - theta2)

    def grad(theta1, theta2, dirac):
        error = sigmoid(logit(theta1, theta2)) - dirac
        return error * grad_u(theta1, theta2)

    true_logits = tf_logits(theta1, theta2)
    sampled_logits = tf_logits(theta3, theta4)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=true_logits, labels=tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=sampled_logits, labels=tf.zeros_like(sampled_logits))
    loss = true_xent + sampled_xent + b
    opt = tf.train.GradientDescentOptimizer(1.0)
    grads_in = opt.compute_gradients(loss, [theta1, theta3])
    grads_out = opt.compute_gradients(loss, [theta2, theta4])
    apply_grad = opt.apply_gradients(grads_in + grads_out)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            b_val = 1.0
            t1, t2, t3, t4 = sess.run([theta1, theta2, theta3, theta4])
            theta1_grad = grad(t1, t2, dirac=1)
            theta3_grad = grad(t3, t4, dirac=0)
            grad_vals_in, grad_vals_out = sess.run([grads_in, grads_out], feed_dict={b: b_val})
            print('theta1 grad should be: ', theta1_grad, 'tf value is: ', grad_vals_in[0][0])
            print('theta3 grad should be: ', theta3_grad, 'tf value is: ', grad_vals_in[1][0])
            print(grad_vals_in)
            print(grad_vals_out)
            sess.run(apply_grad, feed_dict={b: b_val})
            assert (round(grad_vals_in[0][0], 5) == round(theta1_grad, 5))
            assert (round(grad_vals_in[1][0], 5) == round(theta3_grad, 5))


if __name__ == '__main__':
    test_remove_nan_from_index_slices()
    # x = np.array([0.0, 0.999])
    # y = np.array([.5, 0.0])
    # print grad_d_x(x, y)
