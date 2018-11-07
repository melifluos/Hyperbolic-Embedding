import sys
import os

sys.path.append(os.path.join('..', 'src', 'python'))

import utils
import numpy as np
import pandas as pd

df1 = pd.DataFrame(data=np.zeros((4, 3)), index=[1, 2, 3, 4])
df2 = pd.DataFrame(data=np.zeros((4, 3)), index=[1, 5, 3, 6])
df3 = pd.DataFrame(data=np.zeros((2, 3)), index=[2, 4])


def test_subtract_intersection():
    res = utils.subtract_intersection(df1, df2)
    assert res.equals(df3)


def test_make_one_hot():
    arr1 = utils.make_one_hot(np.array([1, 1, 3, 4, 6]))
    assert (np.array_equal(arr1.shape, np.array([5, 6])))
    arr2 = utils.make_one_hot(np.array([0, 1, 1, 3, 4, 6]))
    assert (np.array_equal(arr2.shape, np.array([6, 7])))
    arr3 = utils.make_one_hot(np.array([0, 0, 0, 0]))
    assert (np.array_equal(arr3.shape, np.array([4, 1])))
