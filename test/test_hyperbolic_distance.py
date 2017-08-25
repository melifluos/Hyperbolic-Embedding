"""
unit tests for the hyperbolic distance function
d(x,y) = arcosh(1 + 2||x-y||^2/((1-||x||^2)(1-||y||^2))
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

# vectors for a 3,4,5 triangle scaled to fit into [0,1)
x = np.array([0.3, 0], dtype=np.float32)
y = np.array([0.3, 0.4], dtype=np.float32)
yvec = np.array([[0.3, 0.4], [0.3, 0.2]], dtype=np.float32)


def np_distance(x, y):
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
    arg = 1 + 2 * norm_square / (denom1 * denom2)
    print arg
    return np.arccosh(arg)


def np_squared_pairwise_euclidean_distance(x, y):
    """
    creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
    :param x: first set of vectors of shape (ndata1, ndim)
    :param y: second set of vectors of shape (ndata2, ndim)
    :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
    """
    x = np.array([[2, 0], [3, 0], [4, 0]])
    y = np.array([[1, 0], [2, 4]])
    xnorm = np.linalg.norm(x, axis=1)
    ynorm = np.linalg.norm(y, axis=1)
    dist = np.square(xnorm[:, None]) + np.square(ynorm[None, :]) - 2 * np.matmul(x, y.T)
    return dist


def tf_squared_pairwise_euclidean_distance(x, y):
    """
    creates a matrix of euclidean distances D(i,j) = ||x[i,:] - y[j,:]||
    :param x: first set of vectors of shape (ndata1, ndim)
    :param y: second set of vectors of shape (ndata2, ndim)
    :return: A numpy array of shape (ndata1, ndata2) of pairwise squared distances
    """
    x = np.array([[2, 0], [3, 0], [4, 0]])
    y = np.array([[1, 0], [2, 4]])
    xnorm = np.linalg.norm(x, axis=1)
    ynorm = np.linalg.norm(y, axis=1)
    dist = np.square(xnorm[:, None]) + np.square(ynorm[None, :]) - 2 * np.matmul(x, y.T)
    return dist


def tf_distance(x, y):
    norm_square = tf.square(tf.norm(x - y, axis=0))
    print norm_square
    denom1 = 1 - tf.square(tf.norm(x, axis=0))
    print denom1
    denom2 = 1 - tf.square(tf.norm(y, axis=0))
    print denom2
    arg = 1 + 2 * norm_square / (denom1 * denom2)
    print arg
    return tf.acosh(arg)


def tf_vec_distance(x, y):
    """
    The hyperbolic distance in the Poincare ball with curvature = -1
    :param x: A 1D tensor of shape (1, ndim)
    :param y: A tensor of shape (nsamples, ndim)
    :return: A tensor of shape (1, nsamples)
    """
    if len(y.shape) > 1:
        norm_square = tf.square(tf.norm(x - y, axis=1))
        denom2 = 1 - tf.square(tf.norm(y, axis=1))
    else:
        norm_square = tf.square(tf.norm(x - y, axis=0))
        denom2 = 1 - tf.square(tf.norm(y, axis=0))
    denom1 = 1 - tf.square(tf.norm(x, axis=0))
    arg = 1 + 2 * norm_square / (denom1 * denom2)
    return tf.acosh(arg)


def test_distance(x, y):
    """
    check that the tensorflow distance is the same as the numpy distance
    :param x:
    :param y:
    :return:
    """
    tfx = tf.Variable(x)
    tfy = tf.Variable(y)
    np_dist = np_distance(x, y)
    print 'np distance', np_dist
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_dist = sess.run(tf_vec_distance(tfx, tfy))
        print 'tf distance', tf_dist, 'of shape: ', tf_dist.shape
        assert np.array_equal(np.round(np_dist, 4), np.round(tf_dist, 4))


if __name__ == '__main__':
    # print np_distance(x, yvec)
    print test_distance(x, yvec)
