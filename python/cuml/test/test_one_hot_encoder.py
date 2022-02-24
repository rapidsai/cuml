# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import math

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cudf import DataFrame
from cuml.preprocessing import OneHotEncoder
from cuml.test.utils import stress_param
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder


def from_df_to_numpy(df):
    if isinstance(df, pd.DataFrame):
        return list(zip(*[df[feature] for feature in df.columns]))
    else:
        return list(zip(*[df[feature].values_host for feature in df.columns]))


def _from_df_to_cupy(df):
    """Transform char columns to integer columns, and then create an array"""
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            if isinstance(df, pd.DataFrame):
                df[col] = [ord(c) for c in df[col]]
            else:
                df[col] = [ord(c) for c in df[col].values_host]
    return cp.array(from_df_to_numpy(df))


def _convert_drop(drop):
    if drop is None or drop == 'first':
        return drop
    return [ord(x) if isinstance(x, str) else x for x in drop.values()]


def generate_inputs_from_categories(categories=None,
                                    n_samples=10,
                                    seed=5060,
                                    as_array=False):
    if categories is None:
        if as_array:
            categories = {'strings': list(range(1000, 4000, 3)),
                          'integers': list(range(1000))}
        else:
            categories = {'strings': ['Foo', 'Bar', 'Baz'],
                          'integers': list(range(1000))}

    rd = np.random.RandomState(seed)
    pandas_df = pd.DataFrame({name: rd.choice(cat, n_samples)
                              for name, cat in categories.items()})
    ary = from_df_to_numpy(pandas_df)
    if as_array:
        inp_ary = cp.array(ary)
        return inp_ary, ary
    else:
        df = DataFrame.from_pandas(pandas_df)
        return df, ary


def assert_inverse_equal(ours, ref):
    if isinstance(ours, cp.ndarray):
        cp.testing.assert_array_equal(ours, ref)
    else:
        assert_frame_equal(ours.to_pandas(), ref.to_pandas())


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_vs_skonehot(as_array):
    X = DataFrame({'gender': ['M', 'F', 'F'], 'int': [1, 3, 2]})
    skX = from_df_to_numpy(X)
    if as_array:
        X = _from_df_to_cupy(X)
        skX = cp.asnumpy(X)

    enc = OneHotEncoder(sparse=True)
    skohe = SkOneHotEncoder(sparse=True)

    ohe = enc.fit_transform(X)
    ref = skohe.fit_transform(skX)

    cp.testing.assert_array_equal(ohe.toarray(), ref.toarray())


@pytest.mark.parametrize('drop', [None,
                                  'first',
                                  {'g': 'F', 'i': 3}])
@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_inverse_transform(drop, as_array):
    X = DataFrame({'g': ['M', 'F', 'F'], 'i': [1, 3, 2]})
    if as_array:
        X = _from_df_to_cupy(X)
        drop = _convert_drop(drop)

    enc = OneHotEncoder(drop=drop)
    ohe = enc.fit_transform(X)
    inv = enc.inverse_transform(ohe)

    assert_inverse_equal(inv, X)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_categories(as_array):
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    categories = DataFrame({'chars': ['a', 'b', 'c'], 'int': [0, 1, 2]})
    if as_array:
        X = _from_df_to_cupy(X)
        categories = _from_df_to_cupy(categories).transpose()

    enc = OneHotEncoder(categories=categories, sparse=False)
    ref = cp.array([[1., 0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0., 1.]])
    res = enc.fit_transform(X)
    cp.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
@pytest.mark.filterwarnings("ignore:((.|\n)*)unknown((.|\n)*):UserWarning:"
                            "cuml[.*]")
def test_onehot_fit_handle_unknown(as_array):
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    Y = DataFrame({'chars': ['c', 'b'], 'int': [0, 2]})
    if as_array:
        X = _from_df_to_cupy(X)
        Y = _from_df_to_cupy(Y)

    enc = OneHotEncoder(handle_unknown='error', categories=Y)
    with pytest.raises(KeyError):
        enc.fit(X)

    enc = OneHotEncoder(handle_unknown='ignore', categories=Y)
    enc.fit(X)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_transform_handle_unknown(as_array):
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    Y = DataFrame({'chars': ['c', 'b'], 'int': [0, 2]})
    if as_array:
        X = _from_df_to_cupy(X)
        Y = _from_df_to_cupy(Y)

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


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
@pytest.mark.filterwarnings("ignore:((.|\n)*)unknown((.|\n)*):UserWarning:"
                            "cuml[.*]")
def test_onehot_inverse_transform_handle_unknown(as_array):
    X = DataFrame({'chars': ['a', 'b'], 'int': [0, 2]})
    Y_ohe = cp.array([[0., 0., 1., 0.],
                      [0., 1., 0., 1.]])
    ref = DataFrame({'chars': [None, 'b'], 'int': [0, 2]})
    if as_array:
        X = _from_df_to_cupy(X)
        ref = DataFrame({0: [None, ord('b')], 1: [0, 2]})

    enc = OneHotEncoder(handle_unknown='ignore')
    enc = enc.fit(X)
    df = enc.inverse_transform(Y_ohe)
    assert_inverse_equal(df, ref)


@pytest.mark.parametrize('drop', [None, 'first'])
@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
@pytest.mark.parametrize("n_samples", [10, 1000, 20000, stress_param(250000)])
@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_random_inputs(drop, sparse, n_samples, as_array):
    X, ary = generate_inputs_from_categories(n_samples=n_samples,
                                             as_array=as_array)

    enc = OneHotEncoder(sparse=sparse, drop=drop, categories='auto')
    sk_enc = SkOneHotEncoder(sparse=sparse, drop=drop, categories='auto')
    ohe = enc.fit_transform(X)
    ref = sk_enc.fit_transform(ary)
    if sparse:
        cp.testing.assert_array_equal(ohe.toarray(), ref.toarray())
    else:
        cp.testing.assert_array_equal(ohe, ref)
    inv_ohe = enc.inverse_transform(ohe)
    assert_inverse_equal(inv_ohe, X)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_drop_idx_first(as_array):
    X_ary = [['c', 2, 'a'],
             ['b', 2, 'b']]
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})
    if as_array:
        X = _from_df_to_cupy(X)
        X_ary = cp.asnumpy(X)

    enc = OneHotEncoder(sparse=False, drop='first', categories='auto')
    sk_enc = SkOneHotEncoder(sparse=False, drop='first', categories='auto')
    ohe = enc.fit_transform(X)
    ref = sk_enc.fit_transform(X_ary)
    cp.testing.assert_array_equal(ohe, ref)
    inv = enc.inverse_transform(ohe)
    assert_inverse_equal(inv, X)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_drop_one_of_each(as_array):
    X = DataFrame({'chars': ['c', 'b'], 'int': [2, 2], 'letters': ['a', 'b']})
    drop = dict({'chars': 'b', 'int': 2, 'letters': 'b'})
    X_ary = from_df_to_numpy(X)
    drop_ary = ['b', 2, 'b']
    if as_array:
        X = _from_df_to_cupy(X)
        X_ary = cp.asnumpy(X)
        drop = drop_ary = _convert_drop(drop)

    enc = OneHotEncoder(sparse=False, drop=drop, categories='auto')
    ohe = enc.fit_transform(X)
    print(ohe.dtype)
    ref = SkOneHotEncoder(sparse=False, drop=drop_ary,
                          categories='auto').fit_transform(X_ary)
    cp.testing.assert_array_equal(ohe, ref)
    inv = enc.inverse_transform(ohe)
    assert_inverse_equal(inv, X)


@pytest.mark.parametrize("drop, pattern",
                         [[dict({'chars': 'b'}),
                           '`drop` should have as many columns'],
                          [dict({'chars': 'b', 'int': [2, 0]}),
                           'Trying to drop multiple values'],
                          [dict({'chars': 'b', 'int': 3}),
                           'Some categories [0-9a-zA-Z, ]* were not found'],
                          [DataFrame({'chars': 'b', 'int': 3}),
                           'Wrong input for parameter `drop`.']])
@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_drop_exceptions(drop, pattern, as_array):
    X = DataFrame({'chars': ['c', 'b', 'd'], 'int': [2, 1, 0]})
    if as_array:
        X = _from_df_to_cupy(X)
        drop = _convert_drop(drop) if not isinstance(drop, DataFrame) else drop

    with pytest.raises(ValueError, match=pattern):
        OneHotEncoder(sparse=False, drop=drop).fit(X)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_get_categories(as_array):
    X = DataFrame({'chars': ['c', 'b', 'd'], 'ints': [2, 1, 0]})
    ref = [np.array(['b', 'c', 'd']), np.array([0, 1, 2])]
    if as_array:
        X = _from_df_to_cupy(X)
        ref[0] = np.array([ord(x) for x in ref[0]])

    enc = OneHotEncoder().fit(X)
    cats = enc.categories_

    for i in range(len(ref)):
        np.testing.assert_array_equal(ref[i], cats[i].to_numpy())


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_sparse_drop(as_array):
    X = DataFrame({'g': ['M', 'F', 'F'], 'i': [1, 3, 2], 'l': [5, 5, 6]})
    drop = {'g': 'F', 'i': 3, 'l': 6}

    ary = from_df_to_numpy(X)
    drop_ary = ['F', 3, 6]
    if as_array:
        X = _from_df_to_cupy(X)
        ary = cp.asnumpy(X)
        drop = drop_ary = _convert_drop(drop)

    enc = OneHotEncoder(sparse=True, drop=drop, categories='auto')
    sk_enc = SkOneHotEncoder(sparse=True, drop=drop_ary, categories='auto')
    ohe = enc.fit_transform(X)
    ref = sk_enc.fit_transform(ary)
    cp.testing.assert_array_equal(ohe.toarray(), ref.toarray())


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_categories_shape_mismatch(as_array):
    X = DataFrame({'chars': ['a'], 'int': [0]})
    categories = DataFrame({'chars': ['a', 'b', 'c']})
    if as_array:
        X = _from_df_to_cupy(X)
        categories = _from_df_to_cupy(categories).transpose()

    with pytest.raises(ValueError):
        OneHotEncoder(categories=categories, sparse=False).fit(X)


def test_onehot_category_specific_cases():
    # See this for reasoning: https://github.com/rapidsai/cuml/issues/2690

    # All of these cases use sparse=False, where
    # test_onehot_category_class_count uses sparse=True

    # ==== 2 Rows (Low before High) ====
    example_df = DataFrame()
    example_df["low_cardinality_column"] = ["A"] * 200 + ["B"] * 56
    example_df["high_cardinality_column"] = cp.linspace(0, 255, 256)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit_transform(example_df)

    # ==== 2 Rows (High before Low, used to fail) ====
    example_df = DataFrame()
    example_df["high_cardinality_column"] = cp.linspace(0, 255, 256)
    example_df["low_cardinality_column"] = ["A"] * 200 + ["B"] * 56

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit_transform(example_df)


@pytest.mark.parametrize('total_classes',
                         [np.iinfo(np.uint8).max, np.iinfo(np.uint16).max],
                         ids=['uint8', 'uint16'])
def test_onehot_category_class_count(total_classes: int):
    # See this for reasoning: https://github.com/rapidsai/cuml/issues/2690
    # All tests use sparse=True to avoid memory errors

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    # ==== 2 Rows ====
    example_df = DataFrame()
    example_df["high_cardinality_column"] = cp.linspace(
        0, total_classes - 1, total_classes)
    example_df["low_cardinality_column"] = ["A"] * 200 + ["B"] * (
        total_classes - 200)

    assert (encoder.fit_transform(example_df).shape[1] == total_classes + 2)

    # ==== 3 Rows ====
    example_df = DataFrame()
    example_df["high_cardinality_column"] = cp.linspace(
        0, total_classes - 1, total_classes)
    example_df["low_cardinality_column"] = ["A"] * total_classes
    example_df["med_cardinality_column"] = ["B"] * total_classes

    assert (encoder.fit_transform(example_df).shape[1] == total_classes + 2)

    # ==== N Rows (Even Split) ====
    num_rows = [3, 10, 100]

    for row_count in num_rows:

        class_per_row = int(math.ceil(total_classes / float(row_count))) + 1
        example_df = DataFrame()

        for row_idx in range(row_count):
            example_df[str(row_idx)] = cp.linspace(
                row_idx * class_per_row, ((row_idx + 1) * class_per_row) - 1,
                class_per_row)

        assert (encoder.fit_transform(example_df).shape[1] == class_per_row *
                row_count)


@pytest.mark.parametrize('as_array', [True, False], ids=['cupy', 'cudf'])
def test_onehot_get_feature_names(as_array):
    fruits = ['apple', 'banana', 'strawberry']
    if as_array:
        fruits = [ord(fruit[0]) for fruit in fruits]
    sizes = [0, 1, 2]
    X = DataFrame({'fruits': fruits, 'sizes': sizes})
    if as_array:
        X = _from_df_to_cupy(X)

    enc = OneHotEncoder().fit(X)

    feature_names_ref = ['x0_'+str(fruit) for fruit in fruits] + \
                        ['x1_'+str(size) for size in sizes]
    feature_names = enc.get_feature_names()
    assert np.array_equal(feature_names, feature_names_ref)

    feature_names_ref = ['fruit_'+str(fruit) for fruit in fruits] + \
                        ['size_'+str(size) for size in sizes]
    feature_names = enc.get_feature_names(['fruit', 'size'])
    assert np.array_equal(feature_names, feature_names_ref)
