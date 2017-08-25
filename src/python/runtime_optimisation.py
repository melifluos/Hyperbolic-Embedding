"""
Experiments with optimisations for hyperbolic embeddings
"""

import utils
import pandas as pd
import numpy as np
import datetime
from gensim.models import Word2Vec


def generate_gensim_embeddings(sentences, outpath, size, iter=5, seed=0):
    """
    :param sentences: An iterable of lists of strings
    :param outpath: loction to write the embeddings to
    :return: None
    """
    model = Word2Vec(sentences, size=size, window=5, min_count=0, sg=1, workers=4,
                     iter=iter, hs=0, negative=5, seed=seed)
    model.wv.save_word2vec_format(outpath)


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