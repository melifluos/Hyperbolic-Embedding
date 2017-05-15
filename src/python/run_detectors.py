"""
A module to test the predictive performance of embeddings
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
import utils
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy import stats
import numpy as np
import datetime

classifiers = [
    LogisticRegression(multi_class='ovr', penalty='l2', solver='liblinear', n_jobs=1, max_iter=1000, C=0.005),
    LogisticRegression(multi_class='ovr', penalty='l1', solver='liblinear', n_jobs=1, max_iter=1000, C=0.1),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.0073),
    # SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=2, n_estimators=50, bootstrap=True, criterion='entropy', max_features=0.1,
                           n_jobs=1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers100000 = [
    LogisticRegression(multi_class='ovr', penalty='l2', solver='liblinear', n_jobs=1, max_iter=1000, C=0.005),
    LogisticRegression(multi_class='ovr', penalty='l1', solver='liblinear', n_jobs=1, max_iter=1000, C=0.1),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.0073),
    # SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=50, bootstrap=True, criterion='entropy', max_features=0.1,
                           n_jobs=1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]


def run_detectors(X, y, names, classifiers, n_folds):
    """
    Runs cross-validation for a classifier
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: A tuple of pandas DataFrames containing macro and micro results
    """
    temp = pd.DataFrame(np.zeros(shape=(len(names), n_folds)))
    temp.index = names
    results = (temp, temp.copy())
    for name, detector in zip(names, classifiers):
        y_pred, results = run_cv_pred(X, y, detector, n_folds, name, results)
        print 'running ', name
        # utils.get_metrics(y, y_pred)
    return results


def run_cv_pred(X, y, clf, n_folds, name, results, debug=True):
    """
    Run n-fold cross validation returning a prediction for every row of X
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :param clf: The
    :param n_folds:
    :return:
    """
    # Construct a kfolds object
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    splits = skf.split(X, y)
    y_pred = np.zeros(shape=(len(y), 2))

    # Iterate through folds
    for idx, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X[train_index, :], X[test_index, :]
        assert len(set(train_index).intersection(test_index)) == 0
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train, y_train)
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            preds = clf.predict_proba(X_test)
        except TypeError:
            preds = clf.predict_proba(X_test.todense())
        macro, micro = utils.get_metrics(y[test_index], preds)
        results[0].loc[name, idx] = macro
        results[1].loc[name, idx] = micro
        y_pred[test_index, :] = preds

    # add on training results
    clf.fit(X, y)
    try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
        preds = clf.predict_proba(X)
    except TypeError:
        preds = clf.predict_proba(X.todense())
    macro, micro = utils.get_metrics(y, preds)
    results[0].loc[name, n_folds] = macro
    results[1].loc[name, n_folds] = micro
    # y_pred[test_index] = preds

    return y_pred, results


def run_all_datasets(datasets, y, names, classifiers, n_folds):
    """
    Loop through a list of datasets running potentially numerous classifiers on each
    :param datasets: iterable of numpy (sparse) arrays
    :param y: numpy (sparse) array of shape = (n_data, n_classes) of (n_data, 1)
    :param names: iterable of classifier names
    :param classifiers: A list of intialised scikit-learn compatible classifiers
    :param n_folds:
    :return: A tuple of pandas DataFrames for each dataset containing (macroF1, microF1)
    """
    results = []
    for data in zip(datasets, names):
        temp = run_detectors(data[0], y, data[1], classifiers, n_folds)
        results.append(temp)
    return results


def reduce_features(features):
    """
    Use a pickled rfecv object to reduce the number of embedding features
    :return:
    """
    rfecv = joblib.load('../../local_resources/rfecv.pkl')
    features = rfecv.transform(features)
    return features


def reduce_embedding(embedding):
    data = reduce_features(embedding.values)
    return pd.DataFrame(index=embedding.index, data=data)


def run_experiments(X, y, names, classifiers, n_reps, train_pct):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    temp = pd.DataFrame(np.zeros(shape=(len(names), n_reps)))
    temp.index = names
    results = (temp, temp.copy())
    for name, detector in zip(names, classifiers):
        print 'running ' + str(name) + ' dataset'
        results = evaluate_test_sample(X, y, detector, n_reps, name, results, train_pct)
    return results


def evaluate_test_sample(X, y, clf, nreps, name, results, train_pct):
    """
    Calculate results for this clf at various train / test split percentages
    :param X: features
    :param y: targets
    :param clf: detector
    :param nreps: number of random repetitions
    :param name: name of the detector
    :param results: A tuple of Pandas DataFrames containing (macro, micro) F1 results
    :param train_pct: The percentage of the data used for training
    :return: A tuple of Pandas DataFrames containing (macro, micro) F1 results
    """
    seed = 0
    for rep in range(nreps):
        # setting a random seed will cause the same sample to be generated each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_pct, random_state=seed, stratify=y)
        seed += 1
        clf.fit(X_train, y_train)
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            preds = clf.predict(X_test)
        except TypeError:
            preds = clf.predict(X_test.todense())
        macro, micro = utils.get_metrics(y_test, preds, auc=False)
        results[0].loc[name, rep] = macro
        results[1].loc[name, rep] = micro
    return results


def MF_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    del rf_features.index.name
    emd = pd.read_csv('../../local_resources/roberto_emd.csv', header=None, index_col=0)
    del emd.index.name
    # emd = reduce_embedding(emd)
    # filter the features by customer ID
    temp = rf_features.join(emd[1], how='inner')
    features = temp.drop(1, axis=1)
    # extract the churn target labels
    print 'class distribution', features['target_churned'].value_counts()
    y = features['target_churned'].values.astype(int)
    # remove the labels
    features = features.ix[:, :-4]
    # encode the categoricals
    features['shippingCountry'] = utils.convert_to_other(features['shippingCountry'], pct=0.05, label='Other')
    features = pd.get_dummies(features, columns=['shippingCountry', 'gender'])
    all_feat = features.join(emd)
    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)
    names = np.array([['L2 without MF', 'L1 without MF', 'RF without MF'], ['L2 with MF', 'L1 with MF', 'RF with MF'],
                      ['L2 just MF', 'L1 just MF', 'RF just MF']])
    # names = np.array([['without MF'], ['with MF']])
    n_folds = 20
    results = run_all_datasets([X1, X2, emd.values], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = '../../results/MF/macro_1of100000no_cat' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/MF/micro_1of100000no_cat' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def gensim_train100000_emd_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_train100000.tsv'
    # feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    emd = pd.read_csv('../../local_results/gensim_train_100000.emd', header=None, index_col=0, skiprows=1, sep=" ")
    # emd = pd.read_csv('../../local_results/customer.emd', header=None, index_col=0, skiprows=1, sep=" ")
    features, y = utils.get_classification_xy(rf_features)
    features = features.loc[emd.index, :]
    y = y.loc[emd.index].values
    all_feat = features.join(emd, how='inner')
    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)
    names = np.array(
        [['L2 without emd', 'L1 without emd', 'RF without emd'], ['L2 with emd', 'L1 with emd', 'RF with emd'],
         ['L2 just emd', 'L1 just emd', 'RF just emd']])
    # names = np.array([['without MF'], ['with MF']])
    n_folds = 3
    results = run_all_datasets([X1, X2, emd.values], y, names, classifiers100000, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = '../../results/neural/gensim_train100000' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/neural/gensim_train100000' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def gensim_1in10000_emd_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    emd = pd.read_csv('../../local_results/customer.emd', header=None, index_col=0, skiprows=1, sep=" ")
    features, y = utils.get_classification_xy(rf_features)
    # select only the data points that we have embeddings for
    features = features.loc[emd.index, :]
    y = y.loc[emd.index].values

    all_feat = features.join(emd, how='inner')
    print 'input features shape', all_feat.shape

    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)
    # names = np.array(
    #     [['L2 without emd'], ['L2 with emd']])
    names = np.array(
        [['L2 without emd'], ['L2 with emd'],
         ['L2 just emd']])
    # names = np.array(
    #     [['L2 without emd', 'L1 without emd', 'RF without emd'], ['L2 with emd', 'L1 with emd', 'RF with emd'],
    #      ['L2 just emd', 'L1 just emd', 'RF just emd']])
    # names = np.array([['without MF'], ['with MF']])
    n_folds = 5
    # np.random.seed(42)
    clf = LogisticRegression(multi_class='ovr', penalty='l2', solver='liblinear', n_jobs=1, max_iter=1000, C=0.005)
    df = run_repetitions([X1, X2, emd.values], y, clf, names, reps=10)
    print df
    # results = run_all_datasets([X1, X2], y, names, [clf], n_folds)
    results = run_all_datasets([X1, X2, emd.values], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = '../../results/neural/gensim_1in10000' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/neural/gensim_1in10000' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def debug_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_train100000.tsv'
    features = pd.read_csv(feature_path, sep='\t', index_col=0)
    assert len(features) == len(features.index.unique())
    del features.index.name
    print 'input features shape', features.shape
    features['shippingCountry'] = utils.convert_to_other(features['shippingCountry'], pct=0.05, label='Other')
    features = pd.get_dummies(features, columns=['shippingCountry', 'gender'])
    y = features['target_churned'].astype(int).values
    # remove the labels
    drop_columns = ['target_sales_value', 'target_returned_value', 'target_net_value',
                    'target_churned']
    features = features.drop(drop_columns, axis=1)

    clf = LogisticRegression(multi_class='ovr', penalty='l2', solver='liblinear', n_jobs=1, max_iter=1000, C=0.05)
    X_feat = features.values

    X_feat = scaler.fit_transform(X_feat)
    names = ['with embedding', 'without embedding']
    nreps = 10
    sizes = [16, 32, 128]
    for size in sizes:
        emd = pd.read_csv('../../local_results/gensim_train_100000_size{0}.emd'.format(size), header=None, index_col=0,
                          skiprows=1, sep=" ")
        X_df = features.join(emd, how='left')
        X_df = X_df.fillna(10.0)
        X = X_df.values
        X = scaler.fit_transform(X)
        results = np.zeros(shape=(2, nreps))
        for idx in range(nreps):
            print 'with embeddings'
            msk = np.random.rand(len(features)) < 0.8
            X_train = X[msk, :]
            X_test = X[~msk, :]
            y_train = y[msk]
            y_test = y[~msk]

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
            res = utils.get_metrics(y_test, probs)[0]
            print 'rep{0} '.format(idx), res
            results[0, idx] = res

            print 'without embedding'
            X_train = X_feat[msk, :]
            X_test = X_feat[~msk, :]

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
            res = utils.get_metrics(y_test, probs)[0]
            print 'rep{0} '.format(idx), res
            results[1, idx] = res

        train = []
        mean = results.mean(axis=1)

        clf.fit(X, y)
        probs = clf.predict_proba(X)
        res = utils.get_metrics(y, probs)[0]
        print 'with embedding train ', res
        train.append(res)

        clf.fit(X_feat, y)
        probs = clf.predict_proba(X_feat)
        res = utils.get_metrics(y, probs)[0]
        print 'without embedding train ', res
        train.append(res)

        df = pd.DataFrame(data=results, index=names)
        df['mean'] = mean
        df['train'] = train

        path = '../../results/neural/gensim_train_100000_size{0}_'.format(size) + utils.get_timestamp() + '.csv'
        df.to_csv(path, index=True)
        test = stats.ttest_rel(a=results[0, :], b=results[1, :])
        print 'significance value ', test
        print df


def run_repetitions(data, target, clf, names, reps, train_pct=0.8):
    """
    Run repeated experiments on random train test splits of the data
    :param data: an iterable of numpy arrays
    :param target: a numpy array of target variables
    :param clf: a scikit-learn classifier
    :param names: the names of the data sets. Size should match data
    :param reps: the number of repetitions to run for each dataset
    :param train_pct: the percentage of the data to use for training. The rest will be held out for the test set.
    :return:
    """
    results = np.zeros(shape=(len(data), reps))
    for rep in range(reps):
        msk = np.random.rand(len(target)) < train_pct
        y_train = target[msk]
        y_test = target[~msk]
        for idx, dataset in enumerate(data):
            X_train = dataset[msk, :]
            X_test = dataset[~msk, :]
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
            res = utils.get_metrics(y_test, probs)[0]
            print 'rep{0} '.format(idx), res
            results[idx, rep] = res
    train = []
    mean = results.mean(axis=1)
    for idx, dataset in enumerate(data):
        clf.fit(dataset, target)
        probs = clf.predict_proba(dataset)
        res = utils.get_metrics(target, probs)[0]
        train.append(res)

    df = pd.DataFrame(data=results, index=names)
    df['mean'] = mean
    df['train'] = train

    return df


def debug_1in10000_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_1in10000.tsv'
    features = pd.read_csv(feature_path, sep='\t', index_col=0)
    # assert len(features) == len(features.index.unique())
    # del features.index.name
    # print 'input features shape', features.shape
    # features['shippingCountry'] = utils.convert_to_other(features['shippingCountry'], pct=0.05, label='Other')
    # features = pd.get_dummies(features, columns=['shippingCountry', 'gender'])
    # y = features['target_churned'].astype(int)
    #
    # # remove the labels
    # drop_columns = ['target_sales_value', 'target_returned_value', 'target_net_value',
    #                 'target_churned']
    # features = features.drop(drop_columns, axis=1)

    features, y = utils.get_classification_xy(features)
    emd = pd.read_csv('../../local_results/customer.emd', header=None, index_col=0, skiprows=1, sep=" ")

    y = y.loc[emd.index].values
    features = features.loc[emd.index, :]
    X_df = features.join(emd, how='inner')
    # y = y.values
    # X_df = features.join(emd, how='left')
    # X_df = X_df.fillna(10.0)
    X = X_df.values

    clf = RandomForestClassifier(max_depth=2, n_estimators=50, bootstrap=True, criterion='entropy', max_features=0.1,
                                 max_leaf_nodes=500, n_jobs=1)
    clf = LogisticRegression(multi_class='ovr', penalty='l2', solver='liblinear', n_jobs=1, max_iter=1000, C=0.005)

    X_feat = features.values
    X = scaler.fit_transform(X)
    X_feat = scaler.fit_transform(X_feat)

    names = ['with embedding', 'without embedding']
    nreps = 10
    np.random.seed(42)
    df = run_repetitions([X, X_feat], y, clf, names, nreps)

    # results = np.zeros(shape=(2, nreps))
    # for idx in range(nreps):
    #     print 'with embeddings'
    #     msk = np.random.rand(len(features)) < 0.8
    #     X_train = X[msk, :]
    #     X_test = X[~msk, :]
    #     y_train = y[msk]
    #     y_test = y[~msk]
    #
    #     clf.fit(X_train, y_train)
    #     probs = clf.predict_proba(X_test)
    #     res = utils.get_metrics(y_test, probs)[0]
    #     print 'rep{0} '.format(idx), res
    #     results[0, idx] = res
    #
    #     print 'without embedding'
    #     X_train = X_feat[msk, :]
    #     X_test = X_feat[~msk, :]
    #
    #     clf.fit(X_train, y_train)
    #     probs = clf.predict_proba(X_test)
    #     res = utils.get_metrics(y_test, probs)[0]
    #     print 'rep{0} '.format(idx), res
    #     results[1, idx] = res

    # train = []
    # mean = results.mean(axis=1)
    #
    # clf.fit(X, y)
    # probs = clf.predict_proba(X)
    # res = utils.get_metrics(y, probs)[0]
    # print 'with embedding train ', res
    # train.append(res)
    #
    # clf.fit(X_feat, y)
    # probs = clf.predict_proba(X_feat)
    # res = utils.get_metrics(y, probs)[0]
    # print 'without embedding train ', res
    # train.append(res)
    #
    # df = pd.DataFrame(data=results, index=names)
    # df['mean'] = mean
    # df['train'] = train

    path = '../../results/neural/gensim_1in10000' + utils.get_timestamp() + '.csv'
    df.to_csv(path, index=True)
    # test = stats.ttest_rel(a=results[0, :], b=results[1, :])
    # print 'significance value ', test
    print df


def gensim_1in10000_debug_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    del rf_features.index.name
    print 'input features shape', rf_features.shape
    emd = pd.read_csv('../../local_results/customer.emd', header=None, index_col=0, skiprows=1, sep=" ")
    print 'input embedding shape', rf_features.shape
    # emd = pd.read_csv('../../local_results/customer.emd', header=None, index_col=0, skiprows=1, sep=" ")
    features, y = utils.get_classification_xy(feature_path, emd)
    assert len(features) == features.index.values.unique()
    # all_feat = features.join(emd, how='inner')
    all_feat = features.join(emd)
    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)
    names = np.array(
        [['L2 without emd', 'L1 without emd', 'RF without emd'], ['L2 with emd', 'L1 with emd', 'RF with emd'],
         ['L2 just emd', 'L1 just emd', 'RF just emd']])
    # names = np.array([['without MF'], ['with MF']])
    n_folds = 10
    results = run_all_datasets([X1, X2, emd.values], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = '../../results/neural/gensim_1in10000' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/neural/gensim_1in10000' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def tf_1in10000_emd_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    emd = pd.read_csv('../../local_results/tf.emd', header=None, index_col=0, skiprows=1, sep=" ")
    features, y = utils.get_classification_xy(rf_features)
    features = features.loc[emd.index, :]
    y = y.loc[emd.index].values
    all_feat = features.join(emd, how='inner')
    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)
    names = np.array(
        [['L2 without emd', 'L1 without emd', 'RF without emd'], ['L2 with emd', 'L1 with emd', 'RF with emd'],
         ['L2 just emd', 'L1 just emd', 'RF just emd']])
    n_folds = 10
    results = run_all_datasets([X1, X2, emd.values], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = '../../results/neural/tf_macro_1in10000' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/neural/tf_micro_1in10000' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def tf_train100000_emd_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_train100000.tsv'
    # feature_path = '../../local_resources/features_train100000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    del rf_features.index.name
    emd = pd.read_csv('../../local_results/tf_train_100000.emd', header=None, index_col=0, skiprows=1, sep=" ")
    # emd = pd.read_csv('../../local_results/tf_train_100000.emd', header=None, index_col=0, skiprows=1, sep=" ")
    features, y = utils.get_classification_xy(feature_path, emd)
    all_feat = features.join(emd)
    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)
    names = np.array(
        [['L2 without emd', 'L1 without emd', 'RF without emd'], ['L2 with emd', 'L1 with emd', 'RF with emd'],
         ['L2 just emd', 'L1 just emd', 'RF just emd']])
    n_folds = 10
    results = run_all_datasets([X1, X2, emd.values], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results, n_folds)
    results, tests = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = '../../results/neural/tf_macro_train100000' + utils.get_timestamp() + '.csv'
    micro_path = '../../results/neural/tf_micro_train100000' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def split_embedding_scenario():
    emd1 = pd.read_csv('../../local_results/tf0.emd', header=None, index_col=0, skiprows=1, sep=" ")
    del emd1.index.name
    emd2 = pd.read_csv('../../local_results/tf1.emd', header=None, index_col=0, skiprows=1, sep=" ")
    del emd2.index.name
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    temp1 = emd1.join(rf_features, how='left')
    y_train = temp1['target_churned'].values.astype(int)
    print 'training counts', pd.Series(y_train).value_counts()
    temp2 = emd2.join(rf_features, how='left')
    y_test = temp2['target_churned'].values.astype(int)
    print 'test counts', pd.Series(y_test).value_counts()

    for clf in classifiers:
        clf.fit(emd1.values, y_train)
        preds = clf.predict_proba(emd2.values)
        print len(preds), preds.sum()
        macro, micro = utils.get_metrics(y_test, preds)
        print macro, micro


def split_gensim_embedding_scenario():
    emd1 = pd.read_csv('../../local_results/gensim1_1in10000.emd', header=None, index_col=0, skiprows=1, sep=" ")
    del emd1.index.name
    emd2 = pd.read_csv('../../local_results/gensim2_1in10000.emd', header=None, index_col=0, skiprows=1, sep=" ")
    del emd2.index.name
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    temp1 = emd1.join(rf_features, how='left')
    y_train = temp1['target_churned'].values.astype(int)
    print 'training counts', pd.Series(y_train).value_counts()
    temp2 = emd2.join(rf_features, how='left')
    y_test = temp2['target_churned'].values.astype(int)
    print 'test counts', pd.Series(y_test).value_counts()

    for clf in classifiers:
        clf.fit(emd1.values, y_train)
        preds = clf.predict_proba(emd2.values)
        print len(preds), preds.sum()
        macro, micro = utils.get_metrics(y_test, preds)
        print macro, micro


def initialised_embedding_scenario():
    emd1 = pd.read_csv('../../local_results/tf0.emd', header=None, index_col=0, skiprows=1, sep=" ")
    del emd1.index.name
    emd2 = pd.read_csv('../../local_results/tf_1in10000_init.emd', header=None, index_col=0, skiprows=1, sep=" ")
    del emd2.index.name
    feature_path = '../../local_resources/features_1in10000.tsv'
    rf_features = pd.read_csv(feature_path, sep='\t', index_col=0)
    temp1 = emd1.join(rf_features, how='left')
    y_train = temp1['target_churned'].values.astype(int)
    print 'training counts', pd.Series(y_train).value_counts()
    test_emd = utils.subtract_intersection(emd2, emd1)
    # temp2 = test_emd.join(rf_features, how='left')
    temp2 = emd2.join(rf_features, how='left')
    y_test = temp2['target_churned'].values.astype(int)
    print 'test counts', pd.Series(y_test).value_counts()

    for clf in classifiers:
        clf.fit(emd1.values, y_train)
        preds = clf.predict_proba(emd2.values)[:, 1]
        # preds = clf.predict_proba(test_emd.values)[:, 1]
        print len(preds), preds.sum()
        macro, micro = utils.get_metrics(y_test, preds)
        print macro, micro


def karate_scenario():
    deepwalk_path = 'local_resources/zachary_karate/size8_walks1_len10.emd'

    y_path = 'local_resources/zachary_karate/y.p'
    x_path = 'local_resources/zachary_karate/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=0)

    names = [['logistic'], ['deepwalk']]

    x_deepwalk = utils.read_embedding(deepwalk_path, target)
    # all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)
    X = [x_deepwalk, normalize(x, axis=0)]
    n_folds = 2
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/karate/deepwalk_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/karate/deepwalk_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/karate/deepwalk_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/karate/deepwalk_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


if __name__ == '__main__':
    s = datetime.datetime.now()
    karate_scenario()
    # tf_1in10000_emd_scenario()
    # gensim_train100000_emd_scenario()
    # debug_1in10000_scenario()
    # print 'split embedding'
    # split_gensim_embedding_scenario()
    # print 'initialised embedding'
    # initialised_embedding_scenario()
    print 'ran in {0} s'.format(datetime.datetime.now() - s)
    # print 'non-initialised'
    # split_embedding_scenario()
    # print 'initialised'
    # initialised_embedding_scenario()
