"""
experiments comparing hyperbolic embeddings to other methods using public datasets
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
# import visualisation
import run_detectors
import os
import hyperbolic_embedding as HE
import polar_embedding as PE
import multilabel_detectors as MLD


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


def generate_karate_embedding():
    y_path = '../../local_resources/karate/y.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/run4/'
    walk_path = '../../local_resources/karate/walks1_len10_p1_q1.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=30, skip_window=5, num_pairs=1500,
                    statistics_interval=0.1,
                    initial_learning_rate=1.0, save_path=log_path, epochs=10, concurrent_steps=1)

    path = '../../local_resources/hyperbolic_embeddings/tf_Win_polar' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/karate/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/karate/figs/poincare_polar_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=range(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=range(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/hyperbolic_embeddings/tf_Wout_polar' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    return path


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
    tests[0].to_csv('../../results/karate/pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('../../results/karate/pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/karate/macro' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/karate/micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def generate_blogcatalog_embedding():
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/blogcatalog/y.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/run1/'
    walk_path = '../../local_resources/blogcatalog/p025_q025_d128_walks.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=256, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=1, concurrent_steps=4)

    path = '../../local_resources/blogcatalog/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/blogcatalog/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/blogcatalog/figs/poincare_polar_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/blogcatalog/embeddings/Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    print('blogcatalog embedding generated in: ', datetime.datetime.now() - s)
    return path


def generate_blogcatalog_embedding_small():
    """
    Uses one walk of length 10 per vertex.
    :return:
    """
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/blogcatalog/y.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/run1/'
    walk_path = '../../local_resources/blogcatalog/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=10, concurrent_steps=4)

    path = '../../local_resources/blogcatalog/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/blogcatalog/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/blogcatalog/figs/poincare_polar_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/blogcatalog/embeddings/Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    print('blogcatalog embedding generated in: ', datetime.datetime.now() - s)
    MLD.blogcatalog_scenario(path)
    return path


def generate_blogcatalog_121_embedding():
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/blogcatalog_121_sample/y.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/run1/'
    walk_path = '../../local_resources/blogcatalog_121_sample/walks.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/blogcatalog_121_sample/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/blogcatalog_121_sample/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/blogcatalog_121_sample/figs/poincare_polar_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/blogcatalog_121_sample/embeddings/Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    print('blogcatalog embedding 121 sample generated in: ', datetime.datetime.now() - s)

    MLD.blogcatalog_121_scenario(path)
    return path


def visualise_embedding():
    import visualisation
    path = '../../local_resources/blogcatalog/embeddings/Win_20170515-160129.csv'
    embedding = pd.read_csv(path, index_col=0).values
    y = utils.read_pickle('../../local_resources/blogcatalog/y.p')
    visualisation.plot_poincare_embedding(embedding, y,
                                          '../../results/blogcatalog/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')


if __name__ == '__main__':
    # visualise_embedding()
    generate_blogcatalog_embedding_small()
    # path = generate_blogcatalog_121_embedding()
    # MLD.blogcatalog_scenario(path)
