"""
Sample a multiclass network by only selecting vertices from a few of the classes
"""

import utils
import numpy as np
import pandas as pd


def filter_single_class(labels):
    """
    remove all vertices with multiple memberships
    :return:
    """
    cat_sums = labels.sum(axis=1)
    single_class_vertices = np.where(cat_sums == 1)[0]
    return single_class_vertices


def get_label_distribution(labels):
    label_sums = labels.sum(axis=0)
    print label_sums


def sample_vertices(labels):
    kept_labels = labels[:, 32:]
    # only keep vertices with these labels
    lab_sums = kept_labels.sum(axis=1)
    vertices = np.where(lab_sums > 0)[0]
    temp = labels[vertices, :31]
    # check that none of the returned vertices have any other class labels. They shouldn't as multi-labels have been removed
    assert temp[:].sum() == 0
    return vertices


def sample_graph(X, y, folder):
    single_class_vertices = filter_single_class(y)
    Xsingle = X[single_class_vertices, :]
    Xsingle = Xsingle[:, single_class_vertices]
    ysingle = y[single_class_vertices, :]
    # check single membership
    assert ysingle.sum() == ysingle.shape[0]
    # only keep some of the labels
    vertices = sample_vertices(ysingle)
    Xsamp = Xsingle[vertices, :]
    Xsamp = Xsamp[:, vertices]
    ysamp = ysingle[vertices, :]
    Xout, yout = prune_disconnected(Xsamp, ysamp)

    utils.persist_sparse_data(folder, Xout, yout[:, 32:])
    return Xout, yout[:, 32:]


def prune_disconnected(X, y):
    keep = np.where(X.sum(axis=1) > 0)[0]
    Xkeep = X[keep, :]
    Xkeep = Xkeep[:, keep]
    ykeep = y[keep, :]
    return Xkeep, ykeep


if __name__ == '__main__':
    X = utils.read_pickle('../../local_resources/blogcatalog/X.p')
    y = utils.read_pickle('../../local_resources/blogcatalog/y.p')
    xpath = '../../local_resources/blogcatalog_121_sample/X.p'
    ypath = '../../local_resources/blogcatalog_121_sample/y.p'
    folder = '../../local_resources/blogcatalog_121_sample'
    Xsamp, ysamp = sample_graph(X, y, folder)
    print X.sum()
    print 'number of vertices connected to one or more other vertices: ', sum(Xsamp.sum(axis=1) > 0)
    print 'label distribution: ', ysamp.sum(axis=0)
    print Xsamp.sum()
    print Xsamp.shape
    print ysamp.shape
