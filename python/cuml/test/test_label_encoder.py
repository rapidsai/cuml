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

from cuml.preprocessing.LabelEncoder import LabelEncoder
import cudf
import numpy as np

import pytest
from sklearn.exceptions import NotFittedError


def _df_to_similarity_mat(df):
    arr = df.to_array().reshape(1, -1)
    return np.pad(arr, [(arr.shape[1] - 1, 0), (0, 0)], "edge")


@pytest.mark.parametrize("length", [10, 1000])
@pytest.mark.parametrize("cardinality", [5, 10, 50])
def test_labelencoder_fit_transform(length, cardinality):
    """ Try encoding the entire df
    """
    df = cudf.Series(np.random.choice(cardinality, (length,)))
    encoded = LabelEncoder().fit_transform(df)

    df_arr = _df_to_similarity_mat(df)
    encoded_arr = _df_to_similarity_mat(encoded)
    assert ((encoded_arr == encoded_arr.T) == (df_arr == df_arr.T)).all()


@pytest.mark.parametrize("length", [10, 100, 1000])
@pytest.mark.parametrize("cardinality", [5, 10, 50])
def test_labelencoder_transform(length, cardinality):
    """ Try fitting and then encoding a small subset of the df
    """
    df = cudf.Series(np.random.choice(cardinality, (length,)))
    le = LabelEncoder().fit(df)
    assert le._fitted

    subset = df.iloc[0:df.shape[0] // 2]
    encoded = le.transform(subset)

    subset_arr = _df_to_similarity_mat(subset)
    encoded_arr = _df_to_similarity_mat(encoded)
    assert (
        (encoded_arr == encoded_arr.T) == (subset_arr == subset_arr.T)
    ).all()


def test_labelencoder_unseen():
    """ Try encoding a value that was not present during fitting
    """
    df = cudf.Series(np.random.choice(10, (10,)))
    le = LabelEncoder().fit(df)
    assert le._fitted

    with pytest.raises(KeyError):
        le.transform(cudf.Series([-1]))


def test_labelencoder_unfitted():
    """ Try calling `.transform()` without fitting first
    """
    df = cudf.Series(np.random.choice(10, (10,)))
    le = LabelEncoder()
    assert not le._fitted

    with pytest.raises(NotFittedError):
        le.transform(df)


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
                           use_fit_transform):
    # prepare LabelEncoder
    le = LabelEncoder()
    if use_fit_transform:
        le.fit_transform(orig_label)
    else:
        le.fit(orig_label)
    assert(le._fitted is True)

    # test if inverse_transform is correct
    reverted = le.inverse_transform(ord_label)
    assert(len(reverted) == len(expected_reverted))
    assert(len(reverted)
           == len(reverted[reverted == expected_reverted]))
    # test if correctly raies ValueError
    with pytest.raises(ValueError, match='y contains previously unseen label'):
        le.inverse_transform(bad_ord_label)


def test_unfitted_inverse_transform():
    """ Try calling `.inverse_transform()` without fitting first
    """
    df = cudf.Series(np.random.choice(10, (10,)))
    le = LabelEncoder()
    assert(not le._fitted)

    with pytest.raises(NotFittedError):
        le.transform(df)


@pytest.mark.parametrize("empty, ord_label",
                         [(cudf.Series([]), cudf.Series([2, 1]))])
def test_empty_input(empty, ord_label):
    # prepare LabelEncoder
    le = LabelEncoder()
    le.fit(empty)
    assert(le._fitted is True)

    # test if correctly raies ValueError
    with pytest.raises(ValueError, match='y contains previously unseen label'):
        le.inverse_transform(ord_label)

    # check fit_transform()
    le = LabelEncoder()
    transformed = le.fit_transform(empty)
    assert(le._fitted is True)
    assert(len(transformed) == 0)
