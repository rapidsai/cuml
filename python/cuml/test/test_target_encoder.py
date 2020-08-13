# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cuml.preprocessing.TargetEncoder import TargetEncoder
import cudf
import pandas
import numpy as np
import cupy as cp
from cuml.test.utils import array_equal
import pytest


def test_targetencoder_fit_transform():
    train = cudf.DataFrame({'category': ['a', 'b', 'b', 'a'],
                            'label': [1, 0, 1, 1]})
    encoder = TargetEncoder()
    train_encoded = encoder.fit_transform(train.category, train.label)
    answer = np.array([1., 1., 0., 1.])
    assert array_equal(train_encoded, answer)

    encoder = TargetEncoder()
    encoder.fit(train.category, train.label)
    train_encoded = encoder.transform(train.category)

    assert array_equal(train_encoded, answer)


def test_targetencoder_transform():
    train = cudf.DataFrame({'category': ['a', 'b', 'b', 'a'],
                            'label': [1, 0, 1, 1]})
    test = cudf.DataFrame({'category': ['b', 'b', 'a', 'b']})
    encoder = TargetEncoder()
    encoder.fit_transform(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    answer = np.array([0.5, 0.5, 1., 0.5])
    assert array_equal(test_encoded, answer)

    encoder = TargetEncoder()
    encoder.fit(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    assert array_equal(test_encoded, answer)


@pytest.mark.parametrize('n_samples', [5000, 500000])
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
def test_targetencoder_random(n_samples, dtype):

    x = cp.random.randint(0, 1000, n_samples).astype(dtype)
    y = cp.random.randint(0, 2, n_samples).astype(dtype)
    xt = cp.random.randint(0, 1000, n_samples).astype(dtype)

    encoder = TargetEncoder()
    encoder.fit_transform(x, y)
    test_encoded = encoder.transform(xt)

    df_train = cudf.DataFrame({'x': x, 'y': y})
    dg = df_train.groupby('x', as_index=False).agg({'y': 'mean'})
    df_test = cudf.DataFrame({'x': xt})
    df_test['row_id'] = cp.arange(len(df_test))
    df_test = df_test.merge(dg, on='x', how='left')
    df_test = df_test.sort_values('row_id')
    answer = df_test['y'].fillna(cp.mean(y).item()).values
    assert array_equal(test_encoded, answer)


def test_targetencoder_multi_column():
    """
    Test jointly encoding multiple columns
    """
    train = cudf.DataFrame({'cat_1': ['a', 'b', 'b', 'a', 'a', 'b'],
                            'cat_2': [1, 1, 2, 2, 1, 2],
                            'label': [1, 0, 1, 1, 0, 1]})
    test = cudf.DataFrame({'cat_1': ['b', 'b', 'a', 'b'],
                           'cat_2': [1, 2, 1, 2]})
    encoder = TargetEncoder()
    train_encoded = encoder.fit_transform(train[['cat_1', 'cat_2']],
                                          train.label)
    test_encoded = encoder.transform(test[['cat_1', 'cat_2']])
    train_answer = np.array([2./3, 2./3, 1., 2./3, 2./3, 1.])
    test_answer = np.array([0., 1., 0.5, 1.])
    assert array_equal(train_encoded, train_answer)
    assert array_equal(test_encoded, test_answer)

    encoder = TargetEncoder()
    encoder.fit(train[['cat_1', 'cat_2']], train.label)
    train_encoded = encoder.transform(train[['cat_1', 'cat_2']])
    test_encoded = encoder.transform(test[['cat_1', 'cat_2']])
    assert array_equal(train_encoded, train_answer)
    assert array_equal(test_encoded, test_answer)


def test_targetencoder_newly_encountered():
    """
    Note that there are newly-encountered values in test,
    namely, 'c' and 'd'.
    """
    train = cudf.DataFrame({'category': ['a', 'b', 'b', 'a'],
                            'label': [1, 0, 1, 1]})
    test = cudf.DataFrame({'category': ['c', 'b', 'a', 'd']})
    encoder = TargetEncoder()
    encoder.fit_transform(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    answer = np.array([0.75, 0.5, 1., 0.75])
    assert array_equal(test_encoded, answer)

    encoder = TargetEncoder()
    encoder.fit(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    assert array_equal(test_encoded, answer)


def test_one_category():
    train = cudf.DataFrame({'category': ['a', 'a', 'a', 'a'],
                            'label': [3, 0, 0, 3]})
    test = cudf.DataFrame({'category': ['c', 'b', 'a', 'd']})

    encoder = TargetEncoder()
    train_encoded = encoder.fit_transform(train.category, train.label)
    answer = np.array([1., 2., 2., 1.])
    assert array_equal(train_encoded, answer)

    test_encoded = encoder.transform(test.category)
    answer = np.array([1.5, 1.5, 1.5, 1.5])
    assert array_equal(test_encoded, answer)


def test_targetencoder_pandas():
    """
    Note that there are newly-encountered values in test,
    namely, 'c' and 'd'.
    """
    train = pandas.DataFrame({'category': ['a', 'b', 'b', 'a'],
                              'label': [1, 0, 1, 1]})
    test = pandas.DataFrame({'category': ['c', 'b', 'a', 'd']})
    encoder = TargetEncoder()
    encoder.fit_transform(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    answer = np.array([0.75, 0.5, 1., 0.75])
    assert array_equal(test_encoded, answer)
    print(type(test_encoded))
    assert isinstance(test_encoded, np.ndarray)


def test_targetencoder_numpy():
    """
    Note that there are newly-encountered values in x_test,
    namely, 3 and 4.
    """
    x_train = np.array([1, 2, 2, 1])
    y_train = np.array([1, 0, 1, 1])
    x_test = np.array([1, 2, 3, 4])
    encoder = TargetEncoder()
    encoder.fit_transform(x_train, y_train)
    test_encoded = encoder.transform(x_test)
    answer = np.array([1., 0.5, 0.75, 0.75])
    assert array_equal(test_encoded, answer)
    print(type(test_encoded))
    assert isinstance(test_encoded, np.ndarray)


def test_targetencoder_cupy():
    """
    Note that there are newly-encountered values in x_test,
    namely, 3 and 4.
    """
    x_train = cp.array([1, 2, 2, 1])
    y_train = cp.array([1, 0, 1, 1])
    x_test = cp.array([1, 2, 3, 4])
    encoder = TargetEncoder()
    encoder.fit_transform(x_train, y_train)
    test_encoded = encoder.transform(x_test)
    answer = np.array([1., 0.5, 0.75, 0.75])
    assert array_equal(test_encoded, answer)
    print(type(test_encoded))
    assert isinstance(test_encoded, cp.ndarray)
