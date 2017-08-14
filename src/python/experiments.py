"""
experiments comparing hyperbolic embeddings to other methods using public datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

# Force matplotlib to not use any Xwindows backend. Needed to run on the cluster
matplotlib.use('Agg')
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

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=1.8)]


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
    import visualisation
    y_path = '../../local_resources/karate/y.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/run4/'
    walk_path = '../../local_resources/karate/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
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


def political_blogs_scenario(embedding_path):
    # deepwalk_path = '../../local_resources/hyperbolic_embeddings/tf_test1.csv'

    y_path = '../../local_resources/political_blogs/y.p'
    x_path = '../../local_resources/political_blogs/X.p'
    sizes = [2, 4, 8, 16, 32, 64, 128]
    deepwalk_embeddings = []
    deepwalk_names = []
    dwpath = '../../local_resources/political_blogs/political_blogs'
    for size in sizes:
        path = dwpath + str(size) + '.emd'
        de = pd.read_csv(path, header=None, index_col=0, skiprows=1, sep=" ")
        de.sort_index(inplace=True)
        deepwalk_embeddings.append(de.values)
        deepwalk_names.append(['deepwalk' + str(size)])

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['hyperbolic'], ['logistic']]
    names = deepwalk_names + names

    embedding = pd.read_csv(embedding_path, index_col=0)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    X = deepwalk_embeddings + [embedding.values, normalize(x, axis=0)]
    n_folds = 10
    results = run_detectors.run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('../../results/political_blogs/pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('../../results/political_blogs/pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/political_blogs/macro' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/political_blogs/micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def run_scenario(folder, embedding_path):
    y_path = '../../local_resources/{}/y.p'.format(folder)
    x_path = '../../local_resources/{}/X.p'.format(folder)
    sizes = [2, 4, 8, 16, 32, 64, 128]
    deepwalk_embeddings = []
    deepwalk_names = []
    dwpath = '../../local_resources/{0}/{1}'.format(folder, folder)
    for size in sizes:
        path = dwpath + str(size) + '.emd'
        de = pd.read_csv(path, header=None, index_col=0, skiprows=1, sep=" ")
        de.sort_index(inplace=True)
        deepwalk_embeddings.append(de.values)
        deepwalk_names.append(['deepwalk' + str(size)])

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['hyperbolic'], ['logistic']]
    names = deepwalk_names + names

    embedding = pd.read_csv(embedding_path, index_col=0)
    X = deepwalk_embeddings + [embedding.values, normalize(x, axis=0)]
    n_folds = 10
    results = run_detectors.run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('../../results/{0}/pvalues{1}.csv'.format(folder, utils.get_timestamp()))
    tests[1].to_csv('../../results/{0}/pvalues{1}.csv'.format(folder, utils.get_timestamp()))
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/{0}/macro{1}.csv'.format(folder, utils.get_timestamp())
    micro_path = '../../results/{0}/micro{1}.csv'.format(folder, utils.get_timestamp())
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
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=10, concurrent_steps=16)

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


def run_embedding(folder, run_scenario=True):
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/{}/y.p'.format(folder)
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/run1/'
    walk_path = '../../local_resources/{}/walks_n1_l10.csv'.format(folder)
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/{0}/embeddings/Win_{1}.csv'.format(folder, utils.get_timestamp())

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/{0}/figs/poincare_polar_Win_{1}.pdf'.format(folder,
                                                                                                     utils.get_timestamp()))
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/{0}/figs/poincare_polar_Wout_{1}.pdf'.format(folder,
                                                                                                      utils.get_timestamp()))
    df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/{0}/embeddings/Wout_{1}.csv'.format(folder, utils.get_timestamp()),
        sep=',')
    print('{} embedding generated in: '.format(folder), datetime.datetime.now() - s)
    if run_scenario:
        MLD.run_scenario(folder, path)
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
                    initial_learning_rate=1.0, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/blogcatalog/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/blogcatalog/figs/small_poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
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
                    initial_learning_rate=1.0, save_path=log_path, epochs=1, concurrent_steps=4)

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


def generate_political_blogs_embedding():
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/political_blogs/y.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/run1/'
    walk_path = '../../local_resources/political_blogs/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/political_blogs/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/political_blogs/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/political_blogs/figs/poincare_polar_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/political_blogs/embeddings/Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    print('political blogs sample generated in: ', datetime.datetime.now() - s)

    political_blogs_scenario(path)
    return path


def multiple_embeddings_and_evaluation_scenario():
    names = ['football', 'adjnoun', 'polbooks']
    for name in names:
        run_embedding(name)


def visualise_deepwalk(emb_path, ypath, outfolder):
    import visualisation
    # path = '../../local_resources/blogcatalog/embeddings/Win_20170515-160129.csv'
    embedding = pd.read_csv(emb_path, index_col=0, header=None, skiprows=1, sep=" ").values
    # ypath = '../../local_resources/blogcatalog/y.p'
    y = utils.read_pickle(ypath)
    y = y['cat'].values
    outpath = outfolder + '/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf'
    visualisation.plot_deepwalk_embedding(embedding, y, outpath)


def nips_experiment_runner():
    names = ['football', 'adjnoun', 'polbooks', 'political_blogs', 'karate']
    # names = ['polbooks']
    # names = ['political_blogs', 'karate']

    for name in names:
        embedding_path = run_embedding(name, run_scenario=False)
        mean_path = '../../results/all/{}_means.csv'.format(name)
        error_path = '../../results/all/{}_errors.csv'.format(name)
        means, errors = MLD.run_test_train_split_scenario(name, embedding_path)
        means.to_csv(mean_path)
        errors.to_csv(error_path)


def plot_deepwalk_embeddings():
    """
    plots the 2D deepwalk embeddings
    :return:
    """
    # names = ['football', 'adjnoun', 'polbooks', 'political_blogs', 'karate']
    # names = ['political_blogs', 'karate']
    names = ['karate']
    for name in names:
        emb_path = '../../local_resources/{}/{}2.emd'.format(name, name)
        ypath = '../../local_resources/{}/y.p'.format(name)
        outfolder = '../../local_resources/{}/deepwalk_figs'.format(name)
        visualise_deepwalk(emb_path, ypath, outfolder)


if __name__ == '__main__':
    # generate_karate_embedding()
    nips_experiment_runner()
    # plot_deepwalk_embeddings()
    # nips_experiment_runner()
    # folder = 'karate'
    # y_path = '../../local_resources/{}/y.p'.format(folder)
    # x_path = '../../local_resources/{}/X.p'.format(folder)
    # reps = 2
    # names = ['karate']
    # x, y = utils.read_data(x_path, y_path, threshold=0)
    # run_repetitions(data, y, classifiers[0], names, reps, train_pct=0.8)
    # visualise_embedding()
    # multiple_embeddings_and_evaluation_scenario()
    # embedding_path = '../../local_resources/political_blogs/embeddings/Win_20170517-165831.csv'
    # political_blogs_scenario(embedding_path)
    # generate_political_blogs_embedding()
    # generate_blogcatalog_embedding()
    # visualise_embedding()
    # generate_blogcatalog_embedding_small()
    # path = generate_blogcatalog_embedding()
    # MLD.blogcatalog_scenario(path)
    # karate_test_scenario('../../local_resources/blogcatalog/embeddings/Win_20170515-113351.csv')
