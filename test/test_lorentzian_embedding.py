"""
Tests for calculating lorentzian embeddings in the hyperboloid with tensorflow
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


def minkowski_dot(u, v):
    """
    Minkowski dot product is the same as the Euclidean dot product, but the first element squared is subtracted
    :param u: a vector
    :param v: a vector
    :return: a scalar dot product
    """
    return tf.tensordot(u, v, 1) - 2 * tf.multiply(u[0], v[0])


def minkowski_dist(u, v):
    """
    The distance between two points in Minkowski space
    :param u:
    :param v:
    :return:
    """
    return tf.acosh(-minkowski_dot(u, v))


def project_onto_tangent_space(hyperboloid_point, ambient_gradient):
    """
    project gradients in the ambiant space onto the tangent space
    :param hyperboloid_point: A point on the hyperboloid
    :param ambient_gradient: The gradient to project
    :return:
    """
    return ambient_gradient + minkowski_dot(hyperboloid_point, ambient_gradient) * hyperboloid_point


def exp_map(base, tangent):
    """
    Map a vector 'tangent' from the tangent space at point 'base' onto the manifold.
    """
    norm = tf.sqrt(tf.maximum(minkowski_dot(tangent, tangent), 0))
    if norm == 0:
        return base
    tangent /= norm
    return tf.cosh(norm) * base + tf.sinh(norm) * tangent


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
    em1 = sess.run(exp_map(p1, g1))
    em2 = sess.run(exp_map(p1, g2))
    em3 = sess.run(exp_map(p1, g3))
    # check that the points are on the hyperboloid
    assert round(sess.run(minkowski_dot(em1, em1)), 4) == -1.
    assert round(sess.run(minkowski_dot(em2, em2)), 4) == -1.
    assert round(sess.run(minkowski_dot(em3, em3)), 4) == -1.
    assert em1[0] == em2[0]
    assert em1[1] == -em2[1]
    assert em3[0] > em1[0]
    assert em3[1] > em1[1]

    # consecutive projections don't change values
    # s = tf.constant([math.sqrt(5), 2])  # point on hyperboloid
    # t = tf.constant([6.2, 3.2])  # random grad
    # temp1 = sess.run(project_onto_tangent_space(s, t))
    # temp2 = sess.run(project_onto_tangent_space(s, temp1))
    # assert np.array_equal(temp1, temp2)


def test_minkowski_dot():
    u = tf.constant([1., 0])
    v = tf.constant([1., 0])
    x = tf.constant([10., 0])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert sess.run(minkowski_dot(u, v)) == -1
    assert sess.run(minkowski_dot(v, x)) == -10


def test_minkowski_dist():
    u = tf.constant([1., 0])
    v = tf.constant([1., 0])
    x = tf.constant([10., 0])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert sess.run(minkowski_dist(u, v)) == 0
    assert sess.run(minkowski_dist(v, x)) != 0


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
    print retval
    assert (np.array_equal(retval, true_val))


def test_gradient_transform_matrix():
    """
    grads come in a matrix of shape (n_vars, embedding_dim)
    :return:
    """
    grads = tf.Variable(np.array([[1, 2], [3, 4], [4, 2]]), dtype=tf.int32)
    true_val = np.array([[-1, 2], [-3, 4], [-4, 2]])
    x = np.eye(grads.shape[1])
    x[0, 0] = -1
    T = tf.constant(x, dtype=tf.int32)
    U = tf.matmul(grads, T)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    retval = sess.run(U)
    print retval
    assert (np.array_equal(retval, true_val))


if __name__ == '__main__':
    test_gradient_transform_matrix()
