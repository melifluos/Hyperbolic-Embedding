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
import hyperbolic_cartesian as HCE
import polar_embedding as PE
import multilabel_detectors as MLD
import euclidean_cartesian as EC

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

    path = '../../local_resources/karate/embeddings/tf_Win_polar' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/karate/figs/poincare_polar_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/karate/figs/poincare_polar_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=range(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=range(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/karate/embeddings/tf_Wout_polar' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    return path


def karate_test_scenario(deepwalk_path):
    y_path = '../../local_resources/karate/y.p'
    x_path = '../../local_resources/karate/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['deepwalk'], ['logistic']]

    x_deepwalk = pd.read_csv(deepwalk_path, header=None, index_col=0, skiprows=1, sep=" ")
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
    log_path = '../../local_resources/tf_logs/blogcatalog_polar/'
    walk_path = '../../local_resources/blogcatalog/p025_q025_d128_walks.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=128, embedding_size=size, neg_samples=10, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=0.2, save_path=log_path, epochs=5, concurrent_steps=16)

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


def generate_blogcatalog_cartesian_embedding():
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/blogcatalog/y.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/blogcatalog_cartesian/final_throw1'
    walk_path = '../../local_resources/blogcatalog/p025_q025_d128_walks.csv'
    size = 128  # dimensionality of the embedding
    params = Params(walk_path, batch_size=128, embedding_size=size, neg_samples=64, skip_window=5, num_pairs=1500,
                    statistics_interval=10,
                    initial_learning_rate=0.1, save_path=log_path, epochs=10, concurrent_steps=12)

    path = '../../local_resources/blogcatalog/embeddings/Win_cartesian' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HCE.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/blogcatalog/figs/poincare_Win_cartesian' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/blogcatalog/figs/poincare_Wout_cartesian' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/blogcatalog/embeddings/Wout_cartesian' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    print('blogcatalog cartesian embedding generated in: ', datetime.datetime.now() - s)
    return path


def run_embedding(folder, learning_rate, run_scenario=True, module=HE):
    """
    Generate an embeddings for a given graph
    :param folder: the name of the folder and also the graph
    :param run_scenario: True if cv results are required
    :param module: An alias for the module containing the specific embedding
    :return: the path to the embedding
    """
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/{}/y.p'.format(folder)
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/run1/'
    walk_path = '../../local_resources/{}/walks_n1_l10.csv'.format(folder)
    size = 4  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=learning_rate, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/{0}/embeddings/Win_{1}.csv'.format(folder, utils.get_timestamp())

    embedding_in, embedding_out = module.main(params)

    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/all/embedding_figs/{}_Win_{}.pdf'.format(folder,
                                                                                                  utils.get_timestamp()))
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/all/embedding_figs/{}_Wout_{}.pdf'.format(folder,
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


def generate_blogcatalog_cartesian_embedding_small():
    """
    Uses one walk of length 10 per vertex.
    :return:
    """
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/blogcatalog/y.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/blogcatalog_small/128D'
    walk_path = '../../local_resources/blogcatalog/walks_n1_l10.csv'
    size = 128  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=1.0,
                    initial_learning_rate=0.1, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/blogcatalog/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HCE.main(params)

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
    log_path = '../../local_resources/tf_logs/polblogs/'
    walk_path = '../../local_resources/political_blogs/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=10.0,
                    initial_learning_rate=1.0, save_path=log_path, epochs=5, concurrent_steps=4)

    path = '../../local_resources/political_blogs/embeddings/Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = HCE.main(params)

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


def batch_size_scenario():
    """
    Generate embeddings using different batch sizes for the ~1000 vertex polblogs network
    :return:
    """
    import visualisation
    s = datetime.datetime.now()
    y_path = '../../local_resources/political_blogs/y.p'
    x_path = '../../local_resources/political_blogs/X.p'
    y = utils.read_pickle(y_path)
    log_path = '../../local_resources/tf_logs/polblogs/'
    walk_path = '../../local_resources/political_blogs/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    embeddings = []
    for batch_size in batch_sizes:
        params = Params(walk_path, batch_size=batch_size, embedding_size=size, neg_samples=5, skip_window=5,
                        num_pairs=1500,
                        statistics_interval=10.0,
                        initial_learning_rate=0.1, save_path=log_path, epochs=5, concurrent_steps=4)

        path = '../../local_resources/political_blogs/embeddings/Win_batch_{}_{}.csv'.format(
            batch_size, utils.get_timestamp())

        embedding_in, embedding_out = HCE.main(params)

        visualisation.plot_poincare_embedding(embedding_in, y,
                                              '../../results/political_blogs/figs/poincare_polar_Win_batch_{}_{}.pdf'.format(
                                                  batch_size, utils.get_timestamp()))
        visualisation.plot_poincare_embedding(embedding_out, y,
                                              '../../results/political_blogs/figs/poincare_polar_Wout_batch_{}_{}.pdf'.format(
                                                  batch_size, utils.get_timestamp()))
        df_in = pd.DataFrame(data=embedding_in, index=np.arange(embedding_in.shape[0]))
        df_in.to_csv(path, sep=',')
        df_out = pd.DataFrame(data=embedding_out, index=np.arange(embedding_out.shape[0]))
        df_out.to_csv(
            '../../local_resources/political_blogs/embeddings/Wout_batch_{}_{}.csv'.format(
                batch_size, utils.get_timestamp()),
            sep=',')
        print('political blogs embedding generated in: ', datetime.datetime.now() - s)
        embeddings.append(embedding_in)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [[str(batch_size)] for batch_size in batch_sizes]
    n_folds = 10
    results = run_detectors.run_all_datasets(embeddings, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('../../results/political_blogs/batch_size_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('../../results/political_blogs/batch_size_pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/political_blogs/batch_size_macro' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/political_blogs/batch_size_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)

    return path


def nips_experiment_runner(module, folder, learning_rate):
    """
    runs the experiments on small graphs submitted to NIPS and MLG
    :param module: The module for the relevant type of embeddings eg. HE for Hyperbolic Embedding
    :return: None
    """
    from visualisation import plot_lines_from_df
    names = ['football', 'adjnoun', 'polbooks', 'political_blogs', 'karate']
    # names = ['karate']

    for name in names:
        embedding_path = run_embedding(name, learning_rate, run_scenario=False, module=module)
        mean_path = '../../results/all/{}_means_{}.csv'.format(name, utils.get_timestamp())
        error_path = '../../results/all/{}_errors_{}.csv'.format(name, utils.get_timestamp())
        means, errors = MLD.run_test_train_split_scenario(name, embedding_path)
        means.to_csv(mean_path)
        errors.to_csv(error_path)
        outpath = '../../results/all/lineplots/{}/{}_{}.pdf'.format(folder, name, utils.get_timestamp())
        plot_lines_from_df(name, mean_path, error_path, outpath)
        # plot_lines_from_df(name, means, errors, outpath)


def plot_deepwalk_embeddings():
    """
    plots the 2D deepwalk embeddings
    :return:
    """
    from visualisation import visualise_deepwalk
    # names = ['football', 'adjnoun', 'polbooks', 'political_blogs', 'karate']
    # names = ['political_blogs', 'karate']
    names = ['karate']
    for name in names:
        emb_path = '../../local_resources/{}/{}2.emd'.format(name, name)
        ypath = '../../local_resources/{}/y.p'.format(name)
        outfolder = '../../local_resources/{}/deepwalk_figs'.format(name)
        visualise_deepwalk(emb_path, ypath, outfolder)


def karate_deepwalk_grid_scenario():
    """
    evaluates a grid of embeddings at different sizes, walk lengths and walks per vertex for the karate network.
    Trying to understand why the DeepWalk performance was so poor.
    :return:
    """
    import os
    y_path = '../../local_resources/karate/y.p'
    x_path = '../../local_resources/karate/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    folder = '../../local_resources/karate/gridsearch/'
    names = [[elem] for elem in os.listdir(folder)]

    embeddings = []
    for name in names:
        emb = pd.read_csv(folder + name[0], header=None, index_col=0, skiprows=1, sep=" ")
        emb.sort_index(inplace=True)
        embeddings.append(emb.values)

    names.append(['hyperbolic'])
    hyp_path = '../../local_resources/karate/embeddings/Win_20170808-185202.csv'
    hyp_emb = pd.read_csv(hyp_path, index_col=0)
    embeddings.append(hyp_emb.values)

    n_folds = 10
    results = run_detectors.run_all_datasets(embeddings, y, names, classifiers, n_folds)
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


def blogcatalog_test_scenario(deepwalk_path):
    y_path = '../../local_resources/blogcatalog/y.p'
    x_path = '../../local_resources/blogcatalog/X.p'

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
    tests[0].to_csv('../../results/karate/deepwalk_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('../../results/karate/deepwalk_micro_pvalues' + utils.get_timestamp() + '.csv')
    print('macro', results[0])
    print('micro', results[1])
    macro_path = '../../results/karate/deepwalk_macro' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/karate/deepwalk_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


if __name__ == '__main__':
    path = generate_blogcatalog_cartesian_embedding()
    blogcatalog_test_scenario(path)
    # deepwalk_path = '../../local_resources/karate/karate128.emd'
    # karate_test_scenario(deepwalk_path)
    # generate_karate_embedding()
    # batch_size_scenario()
    # nips_experiment_runner(module=HCE, folder='cartesian', learning_rate=0.2)
    # nips_experiment_runner(module=HE, folder='polar', learning_rate=1)
