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
import cuml
from cuml.dask.preprocessing.LabelEncoder import LabelEncoder
import cudf
import numpy as np
import dask_cudf
import pytest
from cuml.common.exceptions import NotFittedError
import cupy as cp


def _arr_to_similarity_mat(arr):
    arr = arr.reshape(1, -1)
    return np.pad(arr, [(arr.shape[1] - 1, 0), (0, 0)], "edge")


@pytest.mark.parametrize("length", [10, 1000])
@pytest.mark.parametrize("cardinality", [5, 10, 50])
def test_labelencoder_fit_transform(length, cardinality, client):
    """ Try encoding the entire df
    """
    tmp = cudf.Series(np.random.choice(cardinality, (length,)))
    df = dask_cudf.from_cudf(tmp, npartitions=len(client.has_what()))
    encoded = cuml.dask.preprocessing.LabelEncoder().fit_transform(df)

    df_arr = df.compute().to_array()
    df_arr = _arr_to_similarity_mat(df_arr)
    encoder_arr = cp.asnumpy(encoded.compute().to_array())
    encoded_arr = _arr_to_similarity_mat(encoder_arr)
    assert ((encoded_arr == encoded_arr.T) == (df_arr == df_arr.T)).all()


@pytest.mark.parametrize("length", [10, 100, 1000])
@pytest.mark.parametrize("cardinality", [5, 10, 50])
def test_labelencoder_transform(length, cardinality, client):
    """ Try fitting and then encoding a small subset of the df
    """
    tmp = cudf.Series(np.random.choice(cardinality, (length,)))
    df = dask_cudf.from_cudf(tmp, npartitions=len(client.has_what()))
    le = LabelEncoder().fit(df)
    assert le._fitted

    encoded = le.transform(df)

    df_arr = df.compute().to_array()
    df_arr = _arr_to_similarity_mat(df_arr)
    encoder_arr = cp.asnumpy(encoded.compute().to_array())
    encoded_arr = _arr_to_similarity_mat(encoder_arr)
    assert (
        (encoded_arr == encoded_arr.T) == (df_arr == df_arr.T)
    ).all()


def test_labelencoder_unseen(client):
    """ Try encoding a value that was not present during fitting
    """
    df = dask_cudf.from_cudf(cudf.Series(np.random.choice(10, (10,))),
                             npartitions=len(client.has_what()))
    le = LabelEncoder().fit(df)
    assert le._fitted

    with pytest.raises(KeyError):
        tmp = dask_cudf.from_cudf(cudf.Series([-100, -120]),
                                  npartitions=len(client.has_what()))
        le.transform(tmp).compute()


def test_labelencoder_unfitted(client):
    """ Try calling `.transform()` without fitting first
    """
    df = dask_cudf.from_cudf(cudf.Series(np.random.choice(10, (10,))),
                             npartitions=len(client.has_what()))
    le = LabelEncoder()
    with pytest.raises(NotFittedError):
        le.transform(df).compute()


@pytest.mark.parametrize("use_fit_transform", [False, True])
@pytest.mark.parametrize(
        "orig_label, ord_label, expected_reverted, bad_ord_label",
        [(cudf.Series(['a', 'b', 'c']),
          cudf.Series([2, 1, 2, 0]),
          cudf.Series(['c', 'b', 'c', 'a']),
          cudf.Series([-1, 1, 2, 0])),
         (cudf.Series(['Tokyo', 'Paris', 'Austin']),
          cudf.Series([0, 2, 0]),
          cudf.Series(['Austin', 'Tokyo', 'Austin']),
          cudf.Series([0, 1, 2, 3])),
         (cudf.Series(['a', 'b', 'c1']),
          cudf.Series([2, 1]),
          cudf.Series(['c1', 'b']),
          cudf.Series([0, 1, 2, 3])),
         (cudf.Series(['1.09', '0.09', '.09', '09']),
          cudf.Series([0, 1, 2, 3]),
          cudf.Series(['.09', '0.09', '09', '1.09']),
          cudf.Series([0, 1, 2, 3, 4]))])
def test_inverse_transform(orig_label, ord_label,
                           expected_reverted, bad_ord_label,
                           use_fit_transform, client):
    n_workers = len(client.has_what())
    orig_label = dask_cudf.from_cudf(orig_label, npartitions=n_workers)
    ord_label = dask_cudf.from_cudf(ord_label,
                                    npartitions=n_workers)
    expected_reverted = dask_cudf.from_cudf(expected_reverted,
                                            npartitions=n_workers)
    bad_ord_label = dask_cudf.from_cudf(bad_ord_label, npartitions=n_workers)

    # prepare LabelEncoder
    le = LabelEncoder()
    if use_fit_transform:
        le.fit_transform(orig_label)
    else:
        le.fit(orig_label)
    assert(le._fitted is True)

    # test if inverse_transform is correct
    reverted = le.inverse_transform(ord_label)
    reverted = reverted.compute().reset_index(drop=True)
    expected_reverted = expected_reverted.compute()

    assert(len(reverted) == len(expected_reverted))
    assert(len(reverted)
           == len(reverted[reverted == expected_reverted]))
    # test if correctly raies ValueError
    with pytest.raises(ValueError, match='y contains previously unseen label'):
        le.inverse_transform(bad_ord_label).compute()


def test_unfitted_inverse_transform(client):
    """ Try calling `.inverse_transform()` without fitting first
    """
    tmp = cudf.Series(np.random.choice(10, (10,)))
    df = dask_cudf.from_cudf(tmp, npartitions=len(client.has_what()))
    le = LabelEncoder()

    with pytest.raises(NotFittedError):
        le.transform(df)


@pytest.mark.parametrize("empty, ord_label",
                         [(cudf.Series([]), cudf.Series([2, 1]))])
def test_empty_input(empty, ord_label, client):
    # prepare LabelEncoder
    n_workers = len(client.has_what())
    empty = dask_cudf.from_cudf(empty, npartitions=n_workers)
    ord_label = dask_cudf.from_cudf(ord_label, npartitions=n_workers)
    le = LabelEncoder()
    le.fit(empty)
    assert(le._fitted is True)

    # test if correctly raies ValueError
    with pytest.raises(ValueError, match='y contains previously unseen label'):
        le.inverse_transform(ord_label).compute()

    # check fit_transform()
    le = LabelEncoder()
    transformed = le.fit_transform(empty).compute()
    assert(le._fitted is True)
    assert(len(transformed) == 0)


def test_masked_encode(client):
    n_workers = len(client.has_what())
    df = cudf.DataFrame({"filter_col": [1, 1, 2, 3, 1, 1, 1, 1, 6, 5],
                         "cat_col": ['a', 'b', 'c', 'd', 'a',
                                     'a', 'a', 'c', 'b', 'c']})
    ddf = dask_cudf.from_cudf(df, npartitions=n_workers)

    ddf_filter = ddf[ddf["filter_col"] == 1]
    filter_encoded = LabelEncoder().fit_transform(ddf_filter["cat_col"])
    ddf_filter = ddf_filter.assign(filter_encoded=filter_encoded.values)

    encoded_filter = LabelEncoder().fit_transform(ddf["cat_col"])
    ddf = ddf.assign(encoded_filter=encoded_filter.values)

    ddf = ddf[ddf.filter_col == 1]

    assert(ddf.encoded_filter ==
           ddf_filter.filter_encoded).compute().all()
