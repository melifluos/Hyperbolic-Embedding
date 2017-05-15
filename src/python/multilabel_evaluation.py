"""
Here we follow the method for generating F1 scores when each test case can have muliple labels that is described in
http://www.aaai.org/Papers/AAAI/2006/AAAI06-067.pdf
and
Scalable Learning of Collective Behavior Based on Sparse Social Dimensions
For a test case with n labels, they rank the likelihoods of the predictive classes and use the top n and treat it as
n independent binary classification
For a test case with labels
y = [1, 4, 3]
and predictive classes of
pred = 3, 2, 4, 1, ,5
We would have 2 TPs, 1 FP, 2 TN, 1 FN
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

__author__ = 'benchamberlain'


def one_hot_predictions(classes, n_labels):
    """
    Get the indices of the predicted labels in a one hot encoded binary prediction matrix
    :param classes: A numpy array of prediction labels in descending order of probability shape=(n_data, n_classes)
    :param n_labels: A numpy array containing the number of labels assigned to each target shape=(n_data)
    :return: A binary numpy array where A(i,j) = 1 if the ith data point has label j. Shape = (n_data, n_classes)
    """
    # this maybe should be a boolean matrix
    one_hot_preds = np.zeros(shape=classes.shape, dtype=int)
    for idx, count in enumerate(n_labels):
        count = int(count)
        row_indices = [idx] * count
        col_indices = classes[idx, 0:count]
        one_hot_preds[row_indices, col_indices] = 1
    return one_hot_preds


def multilabel_F1(true_mat, pred_mat):
    """
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Macro averaging means that we first calculate prec and rec for each class
    prec = mean(prec_i) where prec_i is the precision for a single class
    rrec = mean(rec_i) where rec_i is the precision for a single class

    Micro averaging means we calculate a single global prec and rec using all
    of the data.
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)


    :param true_mat: A binary matrix of true labels, one column for each label
    :param pred_mat: A binary matrix of predicted labels, one column for each label
    :return:
    """
    fn = get_fn(pred_mat, true_mat)
    print 'fn', fn
    fp = get_fp(pred_mat, true_mat)
    print 'fp', fp
    tp = get_tp(pred_mat, true_mat)
    print 'tp', tp
    macro_prec, micro_prec, class_prec = get_prec(tp, fp)
    print 'macro_prec, micro_prec', macro_prec, micro_prec
    macro_rec, micro_rec, class_rec = get_rec(tp, fn)
    print 'macro_rec, micro_rec', macro_rec, micro_rec
    print 'F1s', np.divide(2 * np.multiply(class_prec, class_rec), class_rec + class_prec)
    macroF1 = get_F1(macro_prec, macro_rec)
    microF1 = get_F1(micro_prec, macro_rec)
    print macroF1, microF1
    return macroF1, microF1


def get_F1(prec, rec):
    """
    get the F1 score
    :param prec:
    :param rec:
    :return:
    """
    return 2 * prec * rec / (prec + rec)


def evaluate(probs, y):
    """
    Evaluate F1 score for multilabel data
    :param y: one hot encoded numpy array of labels
    :param preds: numpy array of predictive probabilities
    :return:
    """
    n_data, n_classes = y.shape
    n_labels = np.array(y.sum(axis=1, dtype=int))
    sorted_class_preds = np.argsort(-probs)
    labels = y.nonzero()
    predictions = one_hot_predictions(sorted_class_preds, n_labels)

    # calculate F1 scores from the two matrices
    macro_res = precision_recall_fscore_support(y, predictions, average='macro')
    # for elem in macro_res:
    #     print elem
    micro_res = precision_recall_fscore_support(y, predictions, average='micro')
    # for elem in micro_res:
    #     print elem
    # macro, micro = multilabel_F1(y, predictions)

    return macro_res[2], micro_res[2]


def get_prec(tp, fp):
    """
    tp / (tp + fp)
    :return:
    """
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    class_precs = np.divide(tp, (tp + fp))
    print 'precs', class_precs
    macro = class_precs.mean()
    micro = tp_sum / (tp_sum + fp_sum)
    return macro, micro, class_precs


def get_rec(tp, fn):
    """
    tp / (tp + fn)
    :return:
    """
    tp_sum = tp.sum()
    fn_sum = fn.sum()
    class_recs = np.divide(tp, (tp + fn))
    print 'recs', class_recs
    macro = class_recs.mean()
    micro = tp_sum / (tp_sum + fn_sum)
    return macro, micro, class_recs


def get_tp(preds, y):
    """
    true positives
    :param preds:
    :param y:
    :return:
    """
    temp = y.multiply(preds)
    return temp.sum(axis=0, dtype=float)


def get_fp(preds, y):
    """
    false positives.
    :param preds:
    :param y:
    :return:
    """
    temp = preds > y
    return temp.sum(axis=0, dtype=float)


def get_fn(preds, y):
    """
    false positives.
    :param preds:
    :param y:
    :return:
    """
    temp = preds < y
    return temp.sum(axis=0, dtype=float)


if __name__ == '__main__':
    preds = np.array([[0, 0, 1, 1, 1],
                      [0, 1, 1, 1, 0],
                      [1, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0]])

    y = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0]])

    multilabel_F1(y, preds)
    res = precision_recall_fscore_support(y, preds, average='macro')
    for elem in res:
        print elem
