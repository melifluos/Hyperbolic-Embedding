"""
utility functions for product based embeddings
"""

from scipy.sparse import lil_matrix
import pandas as pd
import cPickle as pickle
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
import time
import scipy.stats as stats

__author__ = 'benchamberlain'


class MLData:
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def next_batch(self, batch_size):
        """
        sample a batch of data
        """
        n_data, _ = self.features.shape
        idx = np.random.choice(n_data, batch_size)
        target_batch = self.target.eval()[idx, :]
        feature_batch = np.array(self.features[idx, :].todense())
        return feature_batch, target_batch


class MLdataset(object):
    """
    supervised ml data object
    """

    def __init__(self, train, test):
        self.train = train
        self.test = test


def get_metrics(y, probs, verbose=False, auc=False):
    """
    generate metrics to assess the detectors
    :param y: the true labels
    :param probs: Predictive class probabilities or predicted labels
    :param verbose: print metric for each class if True
    :param auc: Calculate auc. If False calculates F1 score
    :return: macro and micro
    """
    if auc:
        try:
            pred = probs[:, 1]
        except IndexError:
            pred = probs
        macro = roc_auc_score(y, pred, average='macro')
        micro = roc_auc_score(y, pred, average='micro')
        all_scores = roc_auc_score(y, pred, average=None)
    else:  # do the F1
        try:
            pred = np.argmax(probs, axis=1)
        except ValueError:
            pred = probs
        macro = f1_score(y, pred, average='macro')
        micro = f1_score(y, pred, average='micro')
        all_scores = f1_score(y, pred, average=None)
    if verbose:
        print 'macro'
        print macro_auc
        print 'micro'
        print micro_auc
        scores = np.zeros(shape=(1, len(all_scores)))
        scores[0, :] = all_scores
        print pd.DataFrame(data=scores, index=None, columns=np.arange(len(all_scores)))
    return macro, micro


def run_cv_pred(X, y, n_folds, model, *args, **kwargs):
    """
    Run n-fold cross validation returning a prediction for every row of X
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :param clf: The
    :param n_folds:
    :return:
    """
    # Construct a kfolds object

    skf = StratifiedKFold(n_splits=n_folds)
    splits = skf.split(X, y)
    y_pred = np.zeros(shape=y.shape)

    # Iterate through folds
    for train_index, test_index in splits:
        test = MLData(X[test_index], y[test_index])
        train = MLData(X[train_index], y[train_index])
        data = MLdataset(train, test)
        # Initialize a classifier with key word arguments
        print('t1')
        model.fit(data)
        print('t2')
        preds = model.predict(data)
        y_pred[test_index] = preds
    return y_pred


def remove_sparse_features(sparse_mat, threshold):
    """
    removes features (stars) with less than threshold observations in this data set
    :param X:
    :param threshold:
    :return: A version of X with columns that are too sparse removed and a list of the good column indices
    """
    print 'input matrix of shape: {0}'.format(sparse_mat.shape)
    observations = np.array(sparse_mat.sum(axis=0)).flatten()
    good_cols = np.where(observations >= threshold)[0]
    out_mat = sparse_mat[:, good_cols]
    print 'output matrix of shape: {0}'.format(out_mat.shape)
    return out_mat, good_cols


def edge_list_to_sparse_mat(edge_list):
    """
    Convert a pandas DF undirected edge list for a bipartite graph into a scipy csc sparse matrix.
    Assumes that edges are contiguosly indexed starting at 0
    :param edge_list: A pandas DF with columns [fan_idx, star_idx]
    :return: A Columnar sparse matrix
    """
    # Create matrix representation (adjacency matrix) of edge list
    data_shape = edge_list.max(axis=0)
    print 'building sparse matrix of size {0}'.format(data_shape)
    X = lil_matrix((data_shape['fan_idx'] + 1, data_shape['star_idx'] + 1), dtype=int)
    X[edge_list['fan_idx'].values, edge_list['star_idx'].values] = 1
    return X.tocsc()


def public_edge_list_to_sparse_mat(edge_list):
    """
    Convert a pandas DF undirected edge list into a scipy csc sparse matrix. Assumes edges are contiguously indexed
    starting at 0
    :param edge_list: A pandas DF with shape (n_data, 2)
    :return: A Columnar sparse matrix
    """
    # Create matrix representation (adjacency matrix) of edge list
    size = edge_list.values[:].max() + 1
    print 'building sparse matrix of size {0}'.format(size, size)
    X = lil_matrix((size, size), dtype=int)
    X[edge_list.ix[:, 0].values, edge_list.ix[:, 1].values] = 1
    X[edge_list.ix[:, 1].values, edge_list.ix[:, 0].values] = 1
    return X.tocsc()


def balance_classes(input_df, n_cat2=23000, n_cat9=1000):
    """
    balances the input data classes so as not to induce incorrect biases in the output
    :param input_df: the raw input data
    :param n_cat2: The number of cat 2 examples to retain
    :param n_older the minimum number of cat 7, 8 and 9 to keep. In reality it might be a bit more as granpeople are
    split over three classes and so making this exact was more trouble than it was worth
    """
    np.random.seed(10)
    cat2 = input_df[input_df['cat'] == 2]
    if len(cat2) > n_cat2:
        rows = np.random.choice(cat2.index.values, n_cat2, replace=False)
        cat2 = cat2.ix[rows]
    cat9 = input_df[input_df['cat'] == 9]
    if len(cat9) > n_cat9:
        rows = np.random.choice(cat9.index.values, n_cat9, replace=False)
        cat9 = cat9.ix[rows]
    input_df = input_df[~input_df['cat'].isin([2, 9])]
    input_df = pd.concat([input_df, cat2, cat9])
    return input_df


def sample_balanced_data(label_path, data_path, outpath, n_samples, max_cat):
    """
    sample n_samples from each decade. Total numbers are:
    1    930006
    2    429184
    3     69976
    4     30663
    5     17307
    6      8010
    7      2870
    8      1612
    9         5
    :param label_path: path to labelled_fans_with_stars if using fan_star_category data or labelled_fans if using
    labelled_fan_friends
    :param data_path path to the data
    :param n_samples: the number of samples to take from each category
    :return:
    """
    data = pd.read_csv(data_path)
    uids = pd.DataFrame(data=data['fan_id'].unique(), columns=['fan_id'])
    # read label information
    labels = pd.read_csv(label_path)
    labels['cat'] = labels['age'].apply(lambda x: int(x / 10))
    # make a greater than final category
    labels['cat1'] = labels.cat.map(lambda x: max_cat if (x > max_cat) else x)
    grouped = labels.groupby('cat1')
    samples = grouped.apply(lambda x: x.sample(n=n_samples))
    samples = samples.reset_index()
    join_df = samples[['fan_id', 'cat1']]
    output = join_df.merge(data)
    output = output.drop('cat', axis=1)
    output = output.rename(columns={'cat1': 'cat'})
    output.to_csv(outpath, index=False)


def remove_duplicate_labelled_fans():
    """
    Creates a deduplicated list of fans from the raw data
    :return:
    """
    fans = pd.read_csv('resources/raw_data/labelled_fans.csv')
    fans = fans.drop_duplicates('fan_id')
    fans[['fan_id', 'age']].to_csv('resources/labelled_fans.csv', index=False)


def preprocess_data(path):
    """
    Reads a csv with columns fan_id star_id star_idx num_followers cat weight
    Removes duplicates and creates and produces data in standard machine learning format X,y
    :param path: path to the training data
    :return: sparse csc matrix X of [fan_idx,star_idx]
    :return: numpy array y of target categories
    """
    temp = pd.read_csv(path, dtype=int)
    input_data = temp.drop_duplicates(['fan_id', 'star_id'])
    # remove known bad IDs
    input_data = remove_bad_ids('resources/exclusion_list.csv', input_data)
    # replace the fan ids with an index
    fan_ids = input_data['fan_id'].drop_duplicates()
    idx = np.arange(len(fan_ids))
    lookup = pd.DataFrame(data={'fan_id': fan_ids.values, 'fan_idx': idx}, index=idx)
    all_data = input_data.merge(lookup, 'left')
    edge_list = all_data[['fan_idx', 'star_idx']]
    edge_list.columns = ['fan_idx', 'star_idx']
    y = all_data[['fan_idx', 'cat']].drop_duplicates()
    X = edge_list_to_sparse_mat(edge_list)
    return X, y, edge_list


def remove_bad_ids(path, data):
    """
    remove ids that have been manually identified as mislabelled
    :param path: path to the list of bad ids
    :param data: a pandas DataFrame containing a 'fan_id' column
    :return: A pandas DataFrame with the bad IDs removed
    """
    exclusion_list = pd.read_csv(path)
    bad_ids = exclusion_list['fan_id']
    data = data[~data['fan_id'].isin(bad_ids)]
    return data


def mat2edgelist(path):
    """
    convert from matlab matrix input types to the edgelist used by the node2vec implementation
    :param mat: matlab matrix type
    :return: pandas dataframe edgelist
    """
    X, _ = read_mat(path)
    indices = X.nonzero()
    data = np.zeros(shape=(len(indices[0]), 2))
    # row vertex index
    data[:, 0] = indices[0]
    # column vertex index
    data[:, 1] = indices[1]
    df = pd.DataFrame(data=data, index=None, columns=['row', 'col'], dtype=int)
    # every edge is counted twice, so only include cases where the row idx is less than the column index
    df = df[df['row'] < df['col']]
    return df


def get_fan_idx_lookup():
    """
    Switch the fan_ids for indices - better for anonymity and making sparse matrics
    :return: writes resources/fan_list.csv
    """
    fans = pd.read_csv('resources/labelled_fans.csv')
    # The duplicates are quite error prone so it is possible to drop them all by setting
    # parameter keep=False this might cause problems with unindexed fans later though.
    fans = fans.drop_duplicates('fan_id')
    fans = fans.reset_index()
    fans[['index', 'fan_id']].to_csv('resources/fan_id2index_lookup.csv', index=False)


def pickle_sparse(sparse, path):
    """
    Writes a sparse matrix to disk in the python cPickle format
    :param sparse: A scipy s
    :param path:
    :return:
    """
    with open(path, 'wb') as outfile:
        pickle.dump(sparse, outfile, protocol=2)


def persist_edgelist(edge_list, path):
    """
    writes the edge_list to file as a .edgelist format file compatible with node2vec
    :param edge_list: A pandas DF with columns [fan_idx, star_idx]
    :param path: the path to write the file to
    :return: None
    """
    edge_list.to_csv(path, index=False, sep=" ", header=False)


def adj2edgelist(adj):
    """
    converts a scipy sparse adjacency matrix to an edglist
    :param adj: a scipy sparse adjacency matrix
    :return: an pandas DF edgelist with columns [fan_idx, star_idx]
    """
    nonzeros = adj.nonzero()
    max_fan_idx = max(nonzeros[0])
    # need to change the indices as the graph is bipartite and otherwise vertices will be interpreted differently
    star_idx = nonzeros[1] + max_fan_idx + 1
    df = pd.DataFrame({'fan_idx': nonzeros[0], 'star_idx': star_idx})
    return df


def prune_disconnected(X, y):
    """
    remove vertices that have no edges from the label and adjacency matrices
    :param X: adj mat
    :param y: labels
    :return:
    """
    keep = np.where(X.sum(axis=1) > 0)[0]
    Xkeep = X[keep, :]
    Xkeep = Xkeep[:, keep]
    try:
        ykeep = y[keep, :]
    except IndexError:
        ykeep = y[keep]
    return Xkeep, ykeep


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def read_gml(inpath, xpath, ypath, attribute_name='value'):
    import networkx as nx
    try:
        G = nx.read_gml(inpath)
    except nx.exception.NetworkXError:
        print """THIS ERROR HAPPENS BECAUSE NETWORKX IS NOT BACKWARD COMPATIBLE AND YOU
        ARE USING AN OLD GML FORMAT. THE BEST OPTION IS TO DOWNGRADE THE VERSION OF NETWORKX
        pip uninstall networkx
        pip install networkx==1.9.1
        """
        raise
    G = G.to_undirected()
    # get labels
    label_dic = nx.get_node_attributes(G, attribute_name)
    y = np.array([v for k, v in label_dic.iteritems()])
    nodes = [k for k, v in label_dic.iteritems()]
    X = nx.adjacency_matrix(G, nodes)
    print('X shape: ', X.shape, 'y shape: ', y.shape)
    # removed as can kill memory
    # assert(check_symmetric(X.todense()))
    X, y = prune_disconnected(X, y)
    print('after pruning disconnected vertices. X shape: ', X.shape, 'y shape: ', y.shape)
    ydf = pd.DataFrame(data=y, index=range(len(y)), columns=['cat'])
    persist_data(xpath, ypath, X, ydf)
    return X, ydf


def persist_data(x_path, y_path, X, y):
    """
    Write the scipy csc sparse matrix X and a pandas DF y to disk
    :param path: the path to write data to
    :param X: scipy sparse css feature matrix
    :param y: pandas DF target values with columns [fan_idx, cat]
    :return: None
    """
    pickle_sparse(X, x_path)
    y.to_pickle(y_path)


def persist_sparse_data(folder, X, y):
    """
    Write the scipy csc sparse matrics X and y to disk. Used for multilabel dataa
    :param path: the path to write data to
    :param X: scipy sparse css feature matrix
    :param y: scipy sparse css multilabel matrix
    :return: None
    """
    pickle_sparse(X, folder + '/X.p')
    pickle_sparse(y, folder + '/y.p')


def read_mat(path):
    """
    Read the .mat files supplied here
    http://leitang.net/social_dimension.html
    :param path: the path to the files
    :return: scipy sparse csc matrices X, y
    """
    data = loadmat(path)
    return data['network'], data['group']


def map_polbooks():
    y = read_pickle('../../local_resources/polbooks/y_char.p')

    def polbook_map(x):
        if x == 'n':
            return 0
        if x == 'c':
            return 1
        if x == 'l':
            return 2

    y['cat'] = y['cat'].apply(polbook_map)
    y.to_pickle('../../local_resources/polbooks/y.p')


def read_roberto_embedding(path, target, size):
    """
    Reads an embedding from text into a matrix
    :param path: the location of the embedding file
    :param size: the number of dimensions of the embedding eg. 64
    :param target: the target variables containing the indices to use
    :return:
    """
    data = pd.read_csv(path, header=None, index_col=0, skiprows=0, names=np.arange(size), sep=" ")
    # make sure the features are in the same order as the targets
    data = data.ix[target['fan_idx']]
    return data.as_matrix()


def read_embedding(path, target, size=None):
    """
    Reads an embedding from text into a matrix
    :param path: the location of the embedding file
    :param size: the number of dimensions of the embedding eg. 64
    :param target: the target variables containing the indices to use
    :return:
    """
    data = pd.read_csv(path, header=None, index_col=0, skiprows=1, sep=" ")
    # make sure the features are in the same order as the targets
    data = data.ix[target['fan_idx']]
    return data.as_matrix()


def read_LINE_embedding(path, target):
    """
    Reads an embedding from text into a matrix
    :param path: the location of the embedding file
    :param size: the number of dimensions of the embedding eg. 64
    :param target: the target variables containing the indices to use
    :return:
    """
    data = pd.read_csv(path, header=None, index_col=0, skiprows=1, sep='\s+')
    # make sure the features are in the same order as the targets
    data = data.ix[target['fan_idx']]
    return data.as_matrix()


def read_tf_embedding(path, target):
    """
    Reads an embedding from text into a matrix
    :param path: the location of the embedding file
    :param size: the number of dimensions of the embedding eg. 64
    :param target: the target variables containing the indices to use
    :return:
    """
    data = pd.read_csv(path, header=None, sep=' ')
    # make sure the features are in the same order as the targets
    data = data.ix[target['fan_idx']]
    return data.as_matrix()


def read_public_embedding(path, size):
    """
    Read the public data sets embeddings files
    :param path:
    :param size:
    :return:
    """
    data = pd.read_csv(path, header=None, index_col=0, skiprows=1, sep=" ")
    # data = pd.read_csv(path, header=None, index_col=0, skiprows=1, names=np.arange(size), sep=" ")
    # make sure the features are in the same order as the targets
    data = data.sort_index(ascending=True)
    return data.as_matrix()


def not_hot(X):
    """
    Take a one hot encoded vector and make it a 1d dense integer vector
    :param X: A sparse one hot encoded matrix
    :return: a 1d numpy array
    """
    return X.nonzero()[1]


def make_one_hot(X):
    """
    Take a vector of integers and convert to one-hot. Assumes min label is zero or 1
    :param X: A 1d numpy integer array of shape (ndata)
    :return: a 2d numpy binary array of shape (ndate, nclasses)
    """
    min = X.min()
    idx = X
    if min == 0:
        nclasses = X.max() + 1
    else:
        nclasses = X.max()
        idx -= 1
    ndata = len(X)
    one_hot = np.zeros(shape=(ndata, nclasses))
    one_hot[range(ndata), X] = 1
    return one_hot


def read_data(x_path, y_path, threshold):
    """
    reads the features and target variables
    :return:
    """
    targets = read_pickle(y_path)
    y = np.array(targets['cat'])
    X = read_pickle(x_path)
    X1, cols = remove_sparse_features(X, threshold=threshold)
    print X1.shape
    return X1, y


def profile_network(folder):
    """
    prints out basic statistics of a network providing y.p and X.p live in folder
    :param folder: path to the data
    :return:
    """
    X, y = read_data(folder + 'X.p', folder + 'y.p', 0)
    print max(y) - min(y) + 1
    print X.sum() / 2
    print np.bincount(y) / float(len(y))


def get_timestamp():
    """
    get a string timestamp to put on files
    :return:
    """
    return time.strftime("%Y%m%d-%H%M%S")


def read_pickle(path):
    with open(path, 'rb') as infile:
        try:
            data = pd.read_pickle(infile)
        except IndexError:  # not a dataframe
            data = pickle.load(infile)
        return data


def read_target(path):
    targets = read_pickle(path)
    targets.cat = targets.cat.astype(int)
    targets.fan_idx = targets.fan_idx.astype(int)
    return targets


def t_grid(results):
    """
    create an all against all grid of significance tests
    :param results:
    :return:
    """
    nrows, ncols = results.shape
    grid = np.zeros((nrows, nrows))
    for row in xrange(nrows):
        for col in xrange(row + 1, nrows):
            test = stats.ttest_rel(a=results.ix[row, 0:-2],
                                   b=results.ix[col, 0:-2])
            grid[row, col] = test.pvalue

    tests = pd.DataFrame(index=results.index, data=grid, columns=results.index)
    return tests


def reshape_res(results):
    macro = []
    micro = []
    for elem in results:
        macro.append(elem[0])
        micro.append(elem[1])
    return macro, micro


def array_t_grid(results, names):
    """
    create an all against all grid of significance tests
    :param results:
    :return:
    """
    nrows = len(results)
    macro_micro = reshape_res(results)
    tests = []
    for elem in macro_micro:
        grid = np.zeros((nrows, nrows))
        for row in xrange(nrows):
            for col in xrange(row + 1, nrows):
                test = stats.ttest_ind(a=elem[row].values,
                                       b=elem[col].values, axis=None, equal_var=False)
                grid[row, col] = test.pvalue

        test = pd.DataFrame(index=names, data=grid, columns=names)
        print test
        tests.append(test)
    return tests


def stats_test(results_tuple):
    """
    performs a 2 sided t-test to see if difference in models is significant
    :param results_tuples: An array of pandas DataFrames (macro,micro)
    :return:
    """
    output = []
    tests = []
    for idx, results in enumerate(results_tuple):
        results['mean'] = results.ix[:, :-1].mean(axis=1)
        results = results.sort_values('mean', ascending=False)

        try:
            print '1 versus 2'
            print(stats.ttest_rel(a=results.ix[0, 0:-2],
                                  b=results.ix[1, 0:-2]))
        except IndexError:
            pass

        try:
            print '2 versus 3'
            print(stats.ttest_rel(a=results.ix[1, 0:-2],
                                  b=results.ix[2, 0:-2]))
        except IndexError:
            pass

        try:
            print '3 versus 4'
            print(stats.ttest_rel(a=results.ix[1, 0:-2],
                                  b=results.ix[2, 0:-2]))
        except IndexError:
            pass

        output.append(results)

        tests.append(t_grid(results))

    return output, tests


def get_names(results_array):
    names = []
    for elem in results_array:
        name = elem[0].index.values[0]
        names.append(name)
    return names


def array_stats_test(results_array):
    """
    performs a 2 sided t-test to see if difference in models is significant. For each condition to be tested the results
    are in a 2d array
    :param results_array: A list of tuples of pandas DataFrames [(macro, micro), (..,..), ...]
    :return:
    """
    names = get_names(results_array)
    output = pd.DataFrame(data=np.zeros(shape=(len(results_array), 2)), index=names,
                          columns=['mean_macro', 'mean_micro'])
    tests = array_t_grid(results_array, names)
    for idx, results in enumerate(results_array):
        output.ix[idx, 0] = results[0].values[:].mean()
        output.ix[idx, 1] = results[1].values[:].mean()
    return output, tests


def preprocess_features(features, other_pct=0.05):
    """
    produce target variable and feature
    :param path: pandas DataFrame of features
    :param other_pct: the threshold relative frequency below which all classes are treated as other
    :return: tuple of pandas DaraFrame (features, labels)
    """
    assert len(features) == len(features.index.unique())
    del features.index.name
    print 'input features shape', features.shape
    # encode the categoricals
    features['shippingCountry'] = convert_to_other(features['shippingCountry'], pct=other_pct, label='Other')
    features = pd.get_dummies(features, columns=['shippingCountry', 'gender'])
    return features


def get_regression_xy(path, emd):
    """
    get features and targets for the LTV regression problem
    :param path: path to raw data
    :param emd: pandas DataFrame of embedding vectors
    :return: tuple of pandas DataFrames (features, targets)
    """
    features = preprocess_features(path, emd)
    features["target_ltv_percentile"] = \
        np.true_divide(stats.rankdata(np.array(features["target_net_value"]), 'min'),
                       len(features["target_net_value"]))
    y = features['target_ltv_percentile'].values
    # remove the labels
    drop_columns = ['target_ltv_percentile', 'target_sales_value', 'target_returned_value', 'target_net_value',
                    'target_churned']
    features = features.drop(drop_columns, axis=1)
    return features, y


def get_classification_xy(features):
    """
    get features and targets for the churn classification problem
    :param features: pandas DataFrame of features
    :return: tuple of pandas DataFrames (features, targets)
    """
    features = preprocess_features(features)
    y = features['target_churned'].astype(int)
    # remove the labels
    drop_columns = ['target_sales_value', 'target_returned_value', 'target_net_value',
                    'target_churned']
    features = features.drop(drop_columns, axis=1)
    return features, y


def convert_to_other(input, pct, label='Other'):
    """
    A function that takes a categorical variable and converts any category below a certain threshold to 'other'
    :param input: A pandas Series containing categorical type data (ints or strings that repeat)
    :param pct: The threshold percentage. Any value appearing less than this will be converted to <label>
    :param label: The label to replace infrequent elements with
    :return: A pandas Series, which is the same size as the input, but with infrequent labels replaced
    """
    size = len(input)
    vc = input.value_counts() / float(size)
    test_case = vc < pct

    def mapper(x):
        output = label
        try:
            if ~test_case[x]:
                # x stays the same
                output = x
        except TypeError:
            pass  # probably got a null, replace with label
        return output

    return input.map(mapper)


def subtract_intersection(df1, df2):
    """
    Return df1 with any rows that have the same index as one in df2 removed
    :param df1: pandas DataFrame
    :param df2: pandas DataFrame
    :return:
    """
    rows, cols = df2.shape
    joindf = pd.DataFrame(index=df2.index, data=np.ones(rows), columns=['dummy'])
    left = df1.join(joindf, how='left')
    filterdf = left[left['dummy'].isnull()]
    filterdf = filterdf.drop('dummy', axis=1)
    return filterdf


def merge_results(results_list, n_folds):
    """
    Take a list of results tuples (macro and micro) and merge into a single tuple
    :param results_list:
    :return: A tuple containing two pandas DataFrames (macro_results, micro_results)
    """
    macro = pd.concat([x[0] for x in results_list])
    micro = pd.concat([x[1] for x in results_list])
    macro.rename(columns={n_folds: 'train'}, inplace=True)
    micro.rename(columns={n_folds: 'train'}, inplace=True)
    return macro, micro


if __name__ == "__main__":
    # X, y, edge_list = preprocess_data('resources/balanced_7class_fan_star_cat.csv')
    # persist_edgelist(edge_list, 'resources/test/balanced7.edgelist')
    # persist_data('resources/test/balanced7X.p', 'resources/test/balanced7y.p',
    #              X, y)
    # path = '../../local_resources/political_blogs/polblogs.gml'
    # ypath = '../../local_resources/political_blogs/y.p'
    # xpath = '../../local_resources/political_blogs/X.p'

    stub = '../../local_resources/'
    names = ['adjnoun', 'football', 'polbooks']
    for name in names:
        path = stub + name + '/' + name + '.gml'
        ypath = stub + name + '/y.p'
        xpath = stub + name + '/X.p'
        print path
        read_gml(path, xpath, ypath, attribute_name='value')

        # adj = read_pickle('resources/test/balanced7_10_thresh_X.p')
        # df = adj2edgelist(adj)
        # persist_edgelist(df, 'resources/test/balanced7_10_thresh.edgelist')

        # edge_list = pd.read_csv('local_resources/zachary_karate/karate.edgelist', header=None)
        # x = public_edge_list_to_sparse_mat(edge_list)
        # y = pd.read_csv('local_resources/zachary_karate/y.csv')
        # persist_data('local_resources/zachary_karate/X.p', 'local_resources/zachary_karate/y.p', x, y)
