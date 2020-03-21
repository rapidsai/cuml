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


def _from_df_to_array(df):
    return list(zip(*[df[feature] for feature in df.columns]))


def test_onehot_vs_skonehot():
    X = DataFrame({'gender': ['Male', 'Female', 'Female'], 'int': [1, 3, 2]})
    skX = _from_df_to_array(X)

    enc = OneHotEncoder(sparse=False)
    skohe = SkOneHotEncoder(sparse=False)

    ohe = enc.fit_transform(X)
    ref = skohe.fit_transform(skX)

    cp.testing.assert_array_equal(ohe, ref)


def test_onehot_inverse_transform():
    X = DataFrame({'gender': ['Male', 'Female', 'Female'], 'int': [1, 3, 2]})

    enc = OneHotEncoder()
    ohe = enc.fit_transform(X)
    inv = enc.inverse_transform(ohe)

    assert X.equals(inv)


def test_onehot_categories():
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    enc = OneHotEncoder(
        categories=DataFrame({'chars': ['a', 'b', 'c'], 'int': [0, 1, 2]}))
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

    enc = OneHotEncoder(handle_unknown='error')
    enc = enc.fit(X)
    with pytest.raises(KeyError):
        enc.transform(Y)

    enc = OneHotEncoder(handle_unknown='ignore')
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
    assert df.equals(ref)


def generate_inputs_from_categories(categories=None,
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


@pytest.mark.parametrize("n_samples", [10, 10000, stress_param(250000)])
def test_onehot_random_inputs(n_samples):
    df, ary = generate_inputs_from_categories(n_samples=n_samples)

    enc = OneHotEncoder(sparse=False)
    sk_enc = SkOneHotEncoder(sparse=False)
    ohe = enc.fit_transform(df)
    ref = sk_enc.fit_transform(ary)
    cp.testing.assert_array_equal(ohe, ref)

    inv_ohe = enc.inverse_transform(ohe)

    assert inv_ohe.equals(df)


def test_onehot_drop_idx_first():
    X_ary = [['c', 2, 'a'],
             ['b', 2, 'b']]
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})

    enc = OneHotEncoder(sparse=False, drop='first')
    sk_enc = SkOneHotEncoder(sparse=False, drop='first')
    ohe = enc.fit_transform(X)
    ref = sk_enc.fit_transform(X_ary)
    cp.testing.assert_array_equal(ohe, ref)


def test_onehot_drop_idx_series():
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})
    drop = dict({'chars': Series(['b']),
                 'int': Series([]),
                 'letters': Series(['a', 'b'])})
    enc = OneHotEncoder(sparse=False, drop=drop)
    ohe = enc.fit_transform(X)
    ref = cp.array([[1., 1.],
                    [0., 1.]])
    cp.testing.assert_array_equal(ohe, ref)


def test_onehot_drop_idx():
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})
    drop = dict({'chars': Series('b'),
                 'int': Series([2]),
                 'letters': Series('b')})
    enc = OneHotEncoder(sparse=False, drop=drop)
    ohe = enc.fit_transform(X)
    ref = SkOneHotEncoder(sparse=False, drop=['b', 2, 'b']).fit_transform(X)
    cp.testing.assert_array_equal(ohe, ref)
