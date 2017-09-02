"""
Experiments with optimisations for hyperbolic embeddings
"""

import utils
import pandas as pd
import numpy as np
import datetime
from gensim.models import Word2Vec
import hyperbolic_cartesian as HCE


def generate_gensim_embeddings(sentences, outpath, params):
    """
    :param sentences: An iterable of lists of strings
    :param outpath: loction to write the embeddings to
    :return: None
    """
    model = Word2Vec(sentences, size=params.embedding_size, window=params.skip_window, min_count=0, sg=1,
                     workers=params.concurrent_steps,
                     iter=params.epochs_to_train, hs=0, negative=params.num_samples, seed=42)
    model.wv.save_word2vec_format(outpath)


def generate_karate_embedding():
    import visualisation
    y_path = '../../local_resources/karate/y.p'
    targets = utils.read_pickle(y_path)
    y = np.array(targets['cat'])
    log_path = '../../local_resources/tf_logs/hyperbolic_cartesian/lr1_epoch1_dim4'
    walk_path = '../../local_resources/karate/walks_n1_l10.csv'
    size = 2  # dimensionality of the embedding
    params = Params(walk_path, batch_size=4, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                    statistics_interval=0.001,
                    initial_learning_rate=0.2, save_path=log_path, epochs=5, concurrent_steps=1)

    path = '../../local_resources/karate/embeddings/hyperbolic_cartesian_Win' + '_' + utils.get_timestamp() + '.csv'

    embedding_in, embedding_out = main(params)
    visualisation.plot_poincare_embedding(embedding_in, y,
                                          '../../results/karate/figs/poincare_Win' + '_' + utils.get_timestamp() + '.pdf')
    visualisation.plot_poincare_embedding(embedding_out, y,
                                          '../../results/karate/figs/poincare_Wout' + '_' + utils.get_timestamp() + '.pdf')
    df_in = pd.DataFrame(data=embedding_in, index=range(embedding_in.shape[0]))
    df_in.to_csv(path, sep=',')
    df_out = pd.DataFrame(data=embedding_out, index=range(embedding_out.shape[0]))
    df_out.to_csv(
        '../../local_resources/karate/embeddings/hyperbolic_cartesian_Wout' + '_' + utils.get_timestamp() + '.csv',
        sep=',')
    return path


def karate_speed_test():
    """
    compare the runtime of cartesian tf embeddings with gensim
    :return:
    """

    walk_path = '../../local_resources/karate/walks_n1_l10.csv'
    outpath = '../../local_results/speedtest/karate_gensim.emd'
    log_path = '../../local_resources/tf_logs/hyperbolic_cartesian/speedtest'
    walks = generate_gensim_sentences(walk_path)
    size = 4  # dimensionality of the embedding
    epochs = 5
    params = HCE.Params(walk_path, batch_size=20, embedding_size=size, neg_samples=5, skip_window=5, num_pairs=1500,
                        statistics_interval=1,
                        initial_learning_rate=0.2, save_path=log_path, epochs=epochs, concurrent_steps=4)

    s = datetime.datetime.now()
    embedding_in, embedding_out = HCE.main(params)
    print 'tf ran in {0} s'.format(datetime.datetime.now() - s)
    s = datetime.datetime.now()
    generate_gensim_embeddings(walks, outpath, params)
    print 'gensim ran in {0} s'.format(datetime.datetime.now() - s)


def blogcatalog_speed_test():
    """
    compare the runtime of cartesian tf embeddings with gensim
    :return:
    """

    walk_path = '../../local_resources/blogcatalog/p025_q025_d128_walks.csv'
    outpath = '../../local_results/speedtest/blogcatalog_gensim.emd'
    log_path = '../../local_resources/tf_logs/hyperbolic_cartesian/speedtest'
    walks = generate_gensim_sentences(walk_path)
    size = 128  # dimensionality of the embedding
    epochs = 5
    params = HCE.Params(walk_path, batch_size=128, embedding_size=size, neg_samples=10, skip_window=5, num_pairs=1500,
                        statistics_interval=5,
                        initial_learning_rate=0.2, save_path=log_path, epochs=epochs, concurrent_steps=4)

    s = datetime.datetime.now()
    embedding_in, embedding_out = HCE.main(params)
    print 'tf ran in {0} s'.format(datetime.datetime.now() - s)
    s = datetime.datetime.now()
    generate_gensim_embeddings(walks, outpath, params)
    print 'gensim ran in {0} s'.format(datetime.datetime.now() - s)


def generate_gensim_sentences(path):
    """
    read data from a product views csv with columns customerId, productId, ... and produce a sentence for each product
    :param gensim: gensim requires that each element is a string and doesn't require indices
    :return: a list of list of customer str(indices)
    """
    walks = pd.read_csv(path, header=None).values
    walks_str = walks.astype(np.str)
    walk_lst = walks_str.tolist()
    return walk_lst


def speed_test_in_paper():
    inpath = '../../local_resources/views1in10000.csv'
    data = pd.read_csv(inpath)
    customers = Customers(data)
    index = customers.reverse_index.customerId
    unigrams = customers.get_unigrams()
    s1 = utils.read_pickle('../../local_results/gensim_sentences_1in10000.pkl')
    s2 = utils.read_pickle('../../local_results/sentences_1in10000.pkl')
    s = datetime.datetime.now()
    generate_gensim_embeddings(s1, '../../local_results/speedtest/gensim.emd', size=64, iter=1, seed=42)
    print 'gensim ran in {0} s'.format(datetime.datetime.now() - s)
    s = datetime.datetime.now()
    generate_tf_embedding_sentences('../../local_results/sentences_1in10000.pkl', index, unigrams, 237376,
                                    '../../local_results/speedtest/tf.emd', size=64,
                                    batch_size=64)
    print 'tf ran in {0} s'.format(datetime.datetime.now() - s)


def speed_test():
    """
    compare tensorflow and gensim runtimes
    :return:
    """
    inpath = '../../local_resources/views1in10000.csv'
    data = pd.read_csv(inpath)
    customers = Customers(data)
    index = customers.reverse_index.customerId
    unigrams = customers.get_unigrams()
    s1 = utils.read_pickle('../../local_results/gensim_sentences_1in10000.pkl')
    s2 = utils.read_pickle('../../local_results/sentences_1in10000.pkl')
    s = datetime.datetime.now()
    generate_gensim_embeddings(s1, '../../local_results/speedtest/gensim.emd', size=64, iter=5, seed=42)
    print 'gensim ran in {0} s'.format(datetime.datetime.now() - s)
    s = datetime.datetime.now()
    generate_custom_tf_embeddings(epochs=5, outpath='../../local_results/speedtest/tf.emd', size=64, batch_size=128,
                                  threads=16)
    print 'tf ran in {0} s'.format(datetime.datetime.now() - s)


if __name__ == '__main__':
    blogcatalog_speed_test()
