"""
unit tests for the components of hyperbolic embeddings in polar co-ordinates.
"""
sys.path.append(os.path.join('..', 'src', 'python'))
import sys
import os
import utils
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from run_detectors import run_all_datasets
import numpy as np
import tensorflow as tf
import math


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_embeddings():
    feature_path = '../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    emd = pd.read_csv('../local_resources/hyperbolic_embeddings/tf_test1.csv', header=None, index_col=0, skiprows=1,
                      sep=" ")
    features, y = utils.get_classification_xy(rf_features)
    features = features.loc[emd.index, :]
    y = y.loc[emd.index].values
    names = np.array([['RF just emd']])
    n_folds = 10
    classifiers = [
        RandomForestClassifier(max_depth=2, n_estimators=50, bootstrap=True, criterion='entropy', max_features=0.1,
                               n_jobs=1)]
    results = run_all_datasets([emd.values], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'tf_testing_1in10000' + utils.get_timestamp() + '.csv'
    micro_path = 'tf_micro_1in10000' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)
    assert results[0]['mean'].values > 0.6


def test_theta_update():
    """
    test that cos(x-y) where x and y are embedding lookups has the correct gradients
    :return:
    """
    x = tf.Variable([0, 1, 2, 3], dtype=tf.float32)
    y = tf.Variable([0, 1, 2, 3], dtype=tf.float32)
    ex1 = tf.placeholder(tf.int32)
    ex2 = tf.placeholder(tf.int32)
    a = tf.nn.embedding_lookup(x, ex1)
    b = tf.nn.embedding_lookup(y, ex2)
    c = tf.cos(tf.subtract(a, b))
    loss = c - 1.0
    opt = tf.train.GradientDescentOptimizer(1.0)
    grad_y = opt.compute_gradients(loss, [y])
    grad_x = opt.compute_gradients(loss, [x])
    apply_grad = opt.apply_gradients(grad_y + grad_x)

    def grad_y_fun(y, yidx, x, xidx):
        print('evaluating grad at y: ', y, 'and index: ', yidx)
        yvals = y[yidx]
        xvals = x[xidx]
        return np.sin(xvals - yvals)

    def check_equivalence(autograd, grad):
        autos = autograd[0][0].values
        for x, y in zip(grad, autos):
            assert (x == y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(20):
            e1 = np.array([1, 1])
            e2 = np.array([2, 3])
            gy, gx, y_val, x_val = sess.run([grad_y, grad_x, y, x], feed_dict={ex1: e1, ex2: e2})
            my_grad = grad_y_fun(y_val, e2, x_val, e1)
            check_equivalence(gy, my_grad)
            # print('grad should be: ', grad_y_fun(y_val, e2, x_val, e1))
            _, cn, yval = sess.run([apply_grad, c, y], feed_dict={ex1: e1, ex2: e2})


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


def test_NCE():
    """
    testing the NCE estimation component in polar co-ordinates
    :return:
    """

    radius_in = tf.Variable([1, 2, 3], dtype=tf.float32,
                            name='radius_in')  # radius
    theta_in = tf.Variable(np.pi * np.array([1, 2, 3]) / 10.0, dtype=tf.float32, name='theta_in')  # angle

    radius_out = tf.Variable([4, 5, 6], dtype=tf.float32,
                             name='radius_out')
    theta_out = tf.Variable(np.pi * np.array([4, 5, 6]), dtype=tf.float32, name='theta_out')
    b = tf.Variable(1.0, name='b')

    example = 0
    label = 2
    sample = 1

    example_radius = tf.nn.embedding_lookup(radius_in, example)
    example_theta = tf.nn.embedding_lookup(theta_in, example)
    sample_radius = tf.nn.embedding_lookup(radius_out, sample)
    sample_theta = tf.nn.embedding_lookup(theta_out, sample)
    label_radius = tf.nn.embedding_lookup(radius_out, label)
    label_theta = tf.nn.embedding_lookup(theta_out, label)

    def tf_logits(r1, r2, theta_in, theta_out):
        cosine = tf.cos(tf.subtract(theta_in, theta_out))
        radius = tf.multiply(r1, r2)
        return tf.multiply(cosine, radius)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def inner_product(r1, r2, theta1, theta2):
        return np.prod([r1, r2, np.cos(theta1 - theta2)])

    def theta_out_grad(r1, r2, theta1, theta2, dirac):
        u = inner_product(r1, r2, theta1, theta2)
        return np.prod([np.sin(theta1 - theta2), r1, (sigmoid(u) - dirac)])

    def modify_grads(grads, radius):
        scaled_grads = []
        for grad, name in grads:
            temp = tf.constant([2.0, 4.0, 8.0], shape=(3, 1))
            temp_indexed = tf.nn.embedding_lookup(temp, grad.indices)
            # g = tf.divide(grad.values, radius[tf.to_int32(grad.indices[0])])
            g = tf.divide(grad.values, temp_indexed)
            # g = tf.divide(grad.values, temp[tf.squeeze(grad.indices[0])])

            # g_clip = tf.clip_by_value(g, -0.1, 0.1)
            scaled_grad = tf.IndexedSlices(g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        # scaled_theta_grad = [(tf.clip_by_value(tf.scatter_div(g, g.indices, radius), -1, 1), v) for g, v in grads]
        return scaled_grads

    def mg(grads, radius):
        scaled_grads = []
        for grad, name in grads:
            g = np.divide(grad.values, radius[grad.indices])
            # g_clip = tf.clip_by_value(g, -0.1, 0.1)
            scaled_grad = (g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        # scaled_theta_grad = [(tf.clip_by_value(tf.scatter_div(g, g.indices, radius), -1, 1), v) for g, v in grads]
        return scaled_grads

    true_logits = tf_logits(example_radius, label_radius, example_theta, label_theta)
    sampled_logits = tf_logits(example_radius, sample_radius, example_theta, sample_theta)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=true_logits, labels=tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=sampled_logits, labels=tf.zeros_like(sampled_logits))
    loss = (tf.reduce_sum(sampled_xent) + tf.reduce_sum(true_xent)) + b
    opt = tf.train.GradientDescentOptimizer(1.0)
    grads_theta_out = opt.compute_gradients(loss, [theta_out])
    print(grads_theta_out)
    indices = grads_theta_out[0][0].indices
    modified_grads = modify_grads(grads_theta_out, radius_out)
    grads = opt.compute_gradients(loss, [radius_in, radius_out, theta_in])
    gv = modified_grads + grads
    # apply_grad = opt.apply_gradients(gv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1):
            b_val = 1.0
            rin, rout, tin, tout = sess.run([radius_in, radius_out, theta_in, theta_out])
            print(rin)
            print(rout)
            print(tin)
            print(tout)
            theta_out_label_grad = theta_out_grad(rin[example], rout[label], tin[example], tout[label], dirac=1)
            theta_out_sample_grad = theta_out_grad(rin[example], rout[sample], tin[example], tout[sample], dirac=0)
            grad_tout, mgrad, indices_val = sess.run([grads_theta_out, modified_grads, indices], feed_dict={b: b_val})
            print('theta out grad: ', grad_tout)
            print('indices are: ', indices_val)
            print('modified theta out grad: ', mgrad)
            print('label grad: ', theta_out_label_grad)
            print('sample grad: ', theta_out_sample_grad)
            print(mg(grad_tout, rout))


def test_modify_grads():
    """
    testing that the gradients of theta can be divided by another tensor
    :return:
    """

    radius_in = tf.Variable([1, 2, 3], dtype=tf.float32,
                            name='radius_in')  # radius
    theta_in = tf.Variable(np.pi * np.array([1, 2, 3]) / 10.0, dtype=tf.float32, name='theta_in')  # angle

    divisor = tf.constant([1, 2, 4], dtype=tf.float32)

    radius_out = tf.Variable([4, 5, 6], dtype=tf.float32,
                             name='radius_out')
    theta_out = tf.Variable(np.pi * np.array([4, 5, 6]), dtype=tf.float32, name='theta_out')
    b = tf.Variable(1.0, name='b')

    example = 0
    label = 2
    sample = 1

    example_radius = tf.nn.embedding_lookup(radius_in, example)
    example_theta = tf.nn.embedding_lookup(theta_in, example)
    sample_radius = tf.nn.embedding_lookup(radius_out, sample)
    sample_theta = tf.nn.embedding_lookup(theta_out, sample)
    label_radius = tf.nn.embedding_lookup(radius_out, label)
    label_theta = tf.nn.embedding_lookup(theta_out, label)

    def tf_logits(r1, r2, theta_in, theta_out):
        cosine = tf.cos(tf.subtract(theta_in, theta_out))
        radius = tf.multiply(r1, r2)
        return tf.multiply(cosine, radius)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def inner_product(r1, r2, theta1, theta2):
        return np.prod([r1, r2, np.cos(theta1 - theta2)])

    def theta_out_grad(r1, r2, theta1, theta2, dirac):
        u = inner_product(r1, r2, theta1, theta2)
        return np.prod([np.sin(theta1 - theta2), r1, (sigmoid(u) - dirac)])

    def modify_grads(grads, radius):
        scaled_grads = []
        for grad, name in grads:
            temp = tf.constant([2.0, 4.0, 8.0], shape=(3, 1))
            temp_indexed = tf.nn.embedding_lookup(radius, grad.indices)
            # g = tf.divide(grad.values, radius[tf.to_int32(grad.indices[0])])
            g = tf.divide(grad.values, temp_indexed)
            # g = tf.divide(grad.values, temp[tf.squeeze(grad.indices[0])])

            # g_clip = tf.clip_by_value(g, -0.1, 0.1)
            scaled_grad = tf.IndexedSlices(g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        # scaled_theta_grad = [(tf.clip_by_value(tf.scatter_div(g, g.indices, radius), -1, 1), v) for g, v in grads]
        return scaled_grads

    def mg(grads, radius):
        scaled_grads = []
        for grad, name in grads:
            g = np.divide(grad.values, radius[grad.indices])
            # g_clip = tf.clip_by_value(g, -0.1, 0.1)
            scaled_grad = (g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        # scaled_theta_grad = [(tf.clip_by_value(tf.scatter_div(g, g.indices, radius), -1, 1), v) for g, v in grads]
        return scaled_grads

    true_logits = tf_logits(example_radius, label_radius, example_theta, label_theta)
    sampled_logits = tf_logits(example_radius, sample_radius, example_theta, sample_theta)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=true_logits, labels=tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=sampled_logits, labels=tf.zeros_like(sampled_logits))
    loss = (tf.reduce_sum(sampled_xent) + tf.reduce_sum(true_xent)) + b
    opt = tf.train.GradientDescentOptimizer(1.0)
    grads_theta_out = opt.compute_gradients(loss, [theta_out])
    print(grads_theta_out)
    indices = grads_theta_out[0][0].indices
    modified_grads = modify_grads(grads_theta_out, radius_out)
    grads = opt.compute_gradients(loss, [radius_in, radius_out, theta_in])
    gv = modified_grads + grads
    apply_grad = opt.apply_gradients(gv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            b_val = 1.0
            rin, rout, tin, tout = sess.run([radius_in, radius_out, theta_in, theta_out])
            print('radius in: ', rin, 'radius out: ', rout, 'theta in:', tin, 'theta out', tout)
            theta_out_label_grad = theta_out_grad(rin[example], rout[label], tin[example], tout[label], dirac=1)
            theta_out_sample_grad = theta_out_grad(rin[example], rout[sample], tin[example], tout[sample], dirac=0)
            [(grad_tout, _)], [(mgrad, _)] = sess.run([grads_theta_out, modified_grads], feed_dict={b: b_val})
            # print('theta out grad: ', grad_tout)
            # print('indices are: ', indices_val)
            mgvals = mgrad.values
            mgindex = mgrad.indices
            # print('modified theta out grad: ', mgrad)
            # print('label grad: ', theta_out_label_grad)
            # print('sample grad: ', theta_out_sample_grad)
            assert (np.round(theta_out_label_grad, 5) == np.round(mgvals[np.where(mgindex == label)], 5))
            assert (np.round(theta_out_sample_grad, 5) == np.round(mgvals[np.where(mgindex == sample)], 5))
            sess.run(apply_grad)
            # print(mg(grad_tout, rout))


def test_gradient_divide_dense():
    """
    test that the divide operator works as expected with gradients descent when using dense tensors
    :return:
    """

    def modify_grads(grads, y):
        scaled_grads = []
        for grad, name in grads:
            g = tf.divide(grad, y)
            scaled_grads.append((g, name))
        return scaled_grads

    x = tf.Variable([2, 3, 4], dtype=tf.float32, name='x')
    y = tf.Variable([2, 4, 8], dtype=tf.float32, name='y')
    b = tf.Variable(1.0, name='b')

    loss = tf.multiply(x, tf.square(y)) - b
    opt = tf.train.GradientDescentOptimizer(1.0)
    x_grad = opt.compute_gradients(loss, [x])
    modified_grads = modify_grads(x_grad, y)
    # modified_grads = x_grad
    grads = opt.compute_gradients(loss, [y])
    gv = modified_grads + grads
    apply_grad = opt.apply_gradients(gv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(2):
            b_val = 1.0
            xval, yval = sess.run([x, y])
            print('x: ', xval, 'y: ', yval)
            [(grad, _)], [(mgrad, _)] = sess.run([x_grad, modified_grads], feed_dict={b: b_val})
            temp = np.divide(grad, yval)
            _ = sess.run([apply_grad], feed_dict={b: b_val})
            # print('check: ', temp)
            # print('grad: ', grad)
            # print('modified grad: ', mgrad)
            assert (np.array_equal(temp, mgrad))
            # print('label grad: ', theta_out_label_grad)
            # print('sample grad: ', theta_out_sample_grad)
            # print(mg(grad_tout, rout))


def test_sparse_index():
    """
    Using the sparse index of a gradient IndexedSlices object to index another tensor
    :return:
    """
    x = tf.Variable([1, 2, 3], dtype=tf.float32)
    y = tf.Variable([2, 4, 6], dtype=tf.float32)
    examples = tf.Variable([0, 1])
    indexed_x = tf.nn.embedding_lookup(x, examples)
    loss = indexed_x
    opt = tf.train.GradientDescentOptimizer(1.0)
    [(grads, name)] = opt.compute_gradients(loss, [x])
    y_indexed = tf.nn.embedding_lookup(y, grads.indices)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        yval = y_indexed.eval()
        assert (np.array_equal(yval, np.array([2., 4.])))


def test_sparse_grad_division():
    """
    Using the sparse index of a gradient IndexedSlices object to index another tensor
    :return:
    """

    def modify_grads(grads, divisor):
        scaled_grads = []
        for grad, name in grads:
            divisor_indexed = tf.nn.embedding_lookup(divisor, grad.indices)
            g = tf.divide(grad.values, divisor_indexed)
            scaled_grad = tf.IndexedSlices(g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        return scaled_grads

    x = tf.Variable([1, 2, 3], dtype=tf.float32)
    y = tf.Variable([2, 4, 6], dtype=tf.float32)
    examples = tf.Variable([0, 1])
    indexed_x = tf.nn.embedding_lookup(x, examples)
    loss = indexed_x
    opt = tf.train.GradientDescentOptimizer(1.0)
    grads = opt.compute_gradients(loss, [x])
    modified_grads = modify_grads(grads, y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [(grads, _)], [(mgrads, _)] = sess.run([grads, modified_grads])
        # print(grads.values)
        # print(mgrads.values)
        assert (np.array_equal(mgrads.values, np.array([0.5, 0.25])))


def test_modified_grad_updates():
    """
    Using modified grads to update variable tensors
    :return:
    """

    def modify_grads(grads, divisor):
        scaled_grads = []
        for grad, name in grads:
            divisor_indexed = tf.nn.embedding_lookup(divisor, grad.indices)
            g = tf.divide(grad.values, divisor_indexed)
            scaled_grad = tf.IndexedSlices(g, grad.indices)
            scaled_grads.append((scaled_grad, name))
        return scaled_grads

    x = tf.Variable([1, 2, 3], dtype=tf.float32)
    y = tf.Variable([2, 4, 6], dtype=tf.float32)
    examples = tf.Variable([0, 1])
    indexed_x = tf.nn.embedding_lookup(x, examples)
    loss = indexed_x
    opt = tf.train.GradientDescentOptimizer(1.0)
    grads = opt.compute_gradients(loss, [x])
    modified_grads = modify_grads(grads, y)
    apply_grads = opt.apply_gradients(modified_grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(3):
            [(gradvals, _)], [(mgradvals, _)], xvals = sess.run([grads, modified_grads, x])
            # print(xvals)
            # print(gradvals)
            # print(mgradvals)
            assert (np.array_equal(mgradvals.values, np.array([0.5, 0.25])))
            _ = sess.run(apply_grads)
            assert (np.array_equal(xvals, np.array([1.0, 2.0, 3.0]) - np.array([.5, .25, 0.]) * epoch))


if __name__ == '__main__':
    test_modify_grads()
    # test_modified_grad_updates()
    # test_sparse_grad_division()
    # test_NCE()
    # test_gradient_divide_dense()
