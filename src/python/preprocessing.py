"""
A pure python module that converts from the long thin files from Azure into a data structure that can be passed to TensorFlow
"""
from collections import defaultdict
import pandas as pd
import numpy as np
import cPickle


class Customers:
    def __init__(self, data):
        self.data = data
        self.index = None
        self.reverse_index = None
        self.construct_indices()

    def construct_indices(self):
        """
        construct indices to get from customerId to idx and back. Used for working with unindexed matrices in TF
        :return:
        """
        uids = self.data.customerId.unique()
        idx = np.arange(len(uids), dtype=int)
        self.index = pd.DataFrame(index=uids, columns=['idx'], data=idx)
        self.reverse_index = pd.DataFrame(index=idx, columns=['customerId'], data=uids)

    def idx2id(self, indices):
        if isinstance(indices, int):
            ids = self.reverse_index.loc[indices, 'customerId']
        else:
            ids = self.reverse_index.loc[indices, 'customerId'].values
        return ids

    def id2idx(self, customerIds):
        if isinstance(customerIds, int):
            indices = self.index.loc[customerIds, 'idx']
        else:
            indices = self.index.loc[customerIds, 'idx'].values
        return indices

    def get_unigrams(self):
        """
        get the distribution of number of product views over customers to be used for negative sampling.
        :return:
        """
        vc = self.data.customerId.value_counts()
        df = self.index.join(vc, how='inner')
        df = df.sort_values('idx', ascending=True)
        return df.customerId.tolist()

    def embed_customers(self, outpath):
        """
        write customer embeddings to file
        :param inpath: the location of the raw product view logs
        :param outpath: location to write the embedding to
        :return:
        """
        sentences, dic = self.generate_sentences()
        generate_embeddings(sentences, outpath)

    def generate_sentences(self, gensim=False, data=None, outpath=None):
        """
        read data from a product views csv with columns customerId, productId, ... and produce a sentence for each product
        :param gensim: gensim requires that each element is a string and doesn't require indices
        :return: a list of list of customer str(indices)
        """
        if not isinstance(data, pd.DataFrame):
            data = self.data
        data = data.set_index('customerId')
        indexed_data = data.join(self.index, how='inner')
        prod_dic = defaultdict(list)
        for index, row in indexed_data.iterrows():
            if gensim:
                cust_id = str(index)  # use the real customer id in string form
            else:
                cust_id = row['idx']  # use an index, which is needed by tensorflow
            prod_dic[row['productId']].append(cust_id)
        # remove products that only a single customer viewed as they contain no embedding information
        prod_dic = {k: v for k, v in prod_dic.iteritems() if len(v) > 1}
        self.prod_dic = prod_dic
        sentences = map(list, prod_dic.values())
        if outpath is not None:
            with open(outpath, 'wb') as f:
                cPickle.dump(sentences, f)
        return sentences


def generate_product_sentences(data):
    """
    generate sequences of products defined by what a single customer viewed
    :param data: raw web logs
    :return:
    """
    cust_dic = defaultdict(list)
    for index, row in data.iterrows():
        cust_dic[row['customerId']].append(str(row['productId']))
    return map(list, cust_dic.values()), cust_dic


def prod2cust_emd(prod_emd, cust_dic):
    """
    for every product a customer viewed add all of the product embeddings together
    :param prod_emd: pandas DF with index of product id
    :param cust_dic: a dictionary with key,val = cust_id, [prod_ids]
    :return:
    """
    emd = np.zeros(shape=(len(cust_dic), prod_emd.shape[1]))
    cust_emd = pd.DataFrame(index=cust_dic.keys(), data=emd)
    for key, val in cust_dic.iteritems():
        product_ids = map(int, val)
        cust_emd.loc[key, :] = prod_emd.loc[product_ids, :].values.sum(axis=0)
    return cust_emd


def read_raw_data():
    path = '../../local_resources/views1in10000.csv'
    data = pd.read_csv(path)
    data = data.sort_values(by='dateViewed')
    return data



if __name__ == '__main__':
    path = '../../local_resources/views1in10000.csv'
    data = pd.read_csv(path)
    customers = Customers(data)
    sentences = customers.generate_sentences(gensim=True, outpath='../../local_results/gensim_sentences_1in10000.pkl')
    # outpaths = ['../../local_results/customer_t1.emd', '../../local_results/customer_t2.emd']
    # inpath = '../../local_resources/views1in10000.csv'
    # embed_split_customers(inpath, outpaths)
    # prod_emd = pd.read_csv('../../local_results/product.emd', header=None, index_col=0, skiprows=1, sep=" ")
    # data = read_raw_data()
    # prod_sentences, cust_dic = generate_product_sentences(data)
    # cust_emd = prod2cust_emd(prod_emd, cust_dic)
    # cust_emd.to_csv('../../local_results/prod2cust_sum.emd')
