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
from cudf import DataFrame
from cuml.preprocessing import OneHotEncoder

import cupy as cp

from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder


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
    with pytest.raises(ValueError):
        enc.fit(X)

    enc = OneHotEncoder(handle_unknown='ignore', categories=Y)
    enc.fit(X)
