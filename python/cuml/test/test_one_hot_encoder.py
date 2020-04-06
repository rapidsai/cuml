# Copyright (c) 2020, NVIDIA CORPORATION.
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
import pytest
from cudf import DataFrame, Series
from cuml.preprocessing import OneHotEncoder

import cupy as cp
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder

from cuml.test.utils import stress_param
from pandas.util.testing import assert_frame_equal


def _from_df_to_array(df):
    return list(zip(*[df[feature] for feature in df.columns]))


def _generate_inputs_from_categories(categories=None,
                                     n_samples=10, seed=5060):
    if categories is None:
        categories = {'strings': ['Foo', 'Bar', 'Baz'],
                      'integers': list(range(1000))}

    rd = np.random.RandomState(seed)
    pandas_df = pd.DataFrame({name: rd.choice(cat, n_samples)
                              for name, cat in categories.items()})
    ary = _from_df_to_array(pandas_df)
    df = DataFrame.from_pandas(pandas_df)
    return df, ary


def test_onehot_vs_skonehot():
    X = DataFrame({'gender': ['Male', 'Female', 'Female'], 'int': [1, 3, 2]})
    skX = _from_df_to_array(X)

    enc = OneHotEncoder(sparse=False)
    skohe = SkOneHotEncoder(sparse=False)

    ohe = enc.fit_transform(X)
    ref = skohe.fit_transform(skX)

    cp.testing.assert_array_equal(ohe, ref)


@pytest.mark.parametrize('drop', [None,
                                  'first',
                                  {'g': Series('F'), 'i': Series(3)}])
def test_onehot_inverse_transform(drop):
    X = DataFrame({'g': ['M', 'F', 'F'], 'i': [1, 3, 2]})

    enc = OneHotEncoder(drop=drop)
    ohe = enc.fit_transform(X)
    inv = enc.inverse_transform(ohe)

    assert_frame_equal(inv.to_pandas(), X.to_pandas())


def test_onehot_categories():
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    enc = OneHotEncoder(
        categories=DataFrame({'chars': ['a', 'b', 'c'], 'int': [0, 1, 2]}),
        sparse=False
    )
    ref = cp.array([[1., 0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0., 1.]])
    res = enc.fit_transform(X)
    cp.testing.assert_array_equal(res, ref)


def test_onehot_fit_handle_unknown():
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    Y = DataFrame({'chars': ['c', 'b'], 'int': [0, 2]})

    enc = OneHotEncoder(handle_unknown='error', categories=Y)
    with pytest.raises(KeyError):
        enc.fit(X)

    enc = OneHotEncoder(handle_unknown='ignore', categories=Y)
    enc.fit(X)


def test_onehot_transform_handle_unknown():
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    Y = DataFrame({'chars': ['c', 'b'], 'int': [0, 2]})

    enc = OneHotEncoder(handle_unknown='error', sparse=False)
    enc = enc.fit(X)
    with pytest.raises(KeyError):
        enc.transform(Y)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(X)
    ohe = enc.transform(Y)
    ref = cp.array([[0., 0., 1., 0.],
                    [0., 1., 0., 1.]])
    cp.testing.assert_array_equal(ohe, ref)


def test_onehot_inverse_transform_handle_unknown():
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    Y_ohe = cp.array([[0., 0., 1., 0.],
                      [0., 1., 0., 1.]])

    enc = OneHotEncoder(handle_unknown='ignore')
    enc = enc.fit(X)
    df = enc.inverse_transform(Y_ohe)
    ref = DataFrame({'chars': [None, 'b'], 'int': [0, 2]})
    assert_frame_equal(df.to_pandas(), ref.to_pandas())


@pytest.mark.parametrize('drop', [None, 'first'])
@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
@pytest.mark.parametrize("n_samples", [10, 10000, 50000, stress_param(250000)])
def test_onehot_random_inputs(drop, sparse, n_samples):
    if sparse:
        pytest.xfail("Sparse arrays are not fully supported by cupy.")

    df, ary = _generate_inputs_from_categories(n_samples=n_samples)

    enc = OneHotEncoder(sparse=sparse, drop=drop)
    sk_enc = SkOneHotEncoder(sparse=sparse, drop=drop)
    ohe = enc.fit_transform(df)
    ref = sk_enc.fit_transform(ary)
    if sparse:
        cp.testing.assert_array_equal(ohe.toarray(), ref.toarray())
    else:
        cp.testing.assert_array_equal(ohe, ref)

    inv_ohe = enc.inverse_transform(ohe)

    assert_frame_equal(inv_ohe.to_pandas(), df.to_pandas())


def test_onehot_drop_idx_first():
    X_ary = [['c', 2, 'a'],
             ['b', 2, 'b']]
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})

    enc = OneHotEncoder(sparse=False, drop='first')
    sk_enc = SkOneHotEncoder(sparse=False, drop='first')
    ohe = enc.fit_transform(X)
    ref = sk_enc.fit_transform(X_ary)
    cp.testing.assert_array_equal(ohe, ref)
    inv = enc.inverse_transform(ohe)
    assert_frame_equal(inv.to_pandas(), X.to_pandas())


def test_onehot_drop_one_of_each():
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})
    drop = dict({'chars': 'b', 'int': 2, 'letters': 'b'})
    enc = OneHotEncoder(sparse=False, drop=drop)
    ohe = enc.fit_transform(X)
    ref = SkOneHotEncoder(sparse=False, drop=['b', 2, 'b']).fit_transform(X)
    cp.testing.assert_array_equal(ohe, ref)
    inv = enc.inverse_transform(ohe)
    assert_frame_equal(inv.to_pandas(), X.to_pandas())


@pytest.mark.parametrize("drop, pattern",
                         [[dict({'chars': 'b'}),
                           '`drop` should have as many columns'],
                          [dict({'chars': 'b', 'int': [2, 0]}),
                           'Trying to drop multiple values'],
                          [dict({'chars': 'b', 'int': 3}),
                           'Some categories [a-zA-Z, ]* were not found'],
                          [DataFrame({'chars': 'b', 'int': 3}),
                           'Wrong input for parameter `drop`.']])
def test_onehot_drop_exceptions(drop, pattern):
    X = DataFrame({'chars': ['c', 'b', 'd'], 'int': [2, 1, 0]})

    with pytest.raises(ValueError, match=pattern):
        OneHotEncoder(sparse=False, drop=drop).fit(X)


def test_onehot_get_categories():
    X = DataFrame({'chars': ['c', 'b', 'd'], 'ints': [2, 1, 0]})

    ref = [np.array(['b', 'c', 'd']), np.array([0, 1, 2])]
    enc = OneHotEncoder().fit(X)
    cats = enc.get_categories_()

    for i in range(len(ref)):
        np.testing.assert_array_equal(ref[i], cats[i])
