#
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
#

import numpy as np
from cudf import Series, DataFrame
import dask_cudf as dc

import pytest

import nvstrings
from cuml.dask.preprocessing import LabelEncoder


@pytest.mark.parametrize(
        "values, keys, expected_encoded, unknown",
        [(dc.from_cudf(Series([2, 1, 3, 1, 3], dtype='int64'), npartitions=2),
          ['1', '2', '3'],
          dc.from_cudf(Series([1, 0, 2, 0, 2], dtype=np.int32), npartitions=2),
          dc.from_cudf(Series([4, 5, 6, 7]), npartitions=2)),
         (dc.from_cudf(Series(['b', 'a', 'c', 'a', 'c'], dtype=object),
                       npartitions=2),
          ['a', 'b', 'c'],
          dc.from_cudf(Series([1, 0, 2, 0, 2], dtype=np.int32), npartitions=2),
          dc.from_cudf(Series(['d', 'e', 'f', 'd']), npartitions=2)),
         (Series(['b', 'a', 'c', 'a', 'c']),
          ['a', 'b', 'c'],
          Series([1, 0, 2, 0, 2], dtype=np.int32),
          Series(['1', '2', '3', '4']))])
def test_fit_and_transform(values, keys, expected_encoded, unknown):
    # Test LabelEncoder's fit and it's reaction to unkown label
    le = LabelEncoder()
    le.fit(values)
    assert(le._fitted is True)

    keys = nvstrings.to_device(keys)
    assert(le._cats.keys().to_host() == keys.to_host())

    # Test LabelEncoder's transform
    encoded = le.transform(values)
    assert(len(encoded) == len(expected_encoded))
    assert(len(encoded) == len(encoded[encoded == expected_encoded]))

    with pytest.raises(ValueError, match='contains previously unseen labels'):
        le.transform(unknown)


@pytest.mark.parametrize(
        "values, keys, expected_encoded, unknown",
        [(dc.from_cudf(Series([2, 1, 3, 1, 3], dtype='int64'), npartitions=2),
          ['1', '2', '3'],
          dc.from_cudf(Series([1, 0, 2, 0, 2], dtype=np.int32), npartitions=2),
          dc.from_cudf(Series([4, 5, 6, 7]), npartitions=2)),
         (dc.from_cudf(Series(['b', 'a', 'c', 'a', 'c'], dtype=object),
                       npartitions=2),
          ['a', 'b', 'c'],
          dc.from_cudf(Series([1, 0, 2, 0, 2], dtype=np.int32), npartitions=2),
          dc.from_cudf(Series(['d', 'e', 'f', 'd']), npartitions=2)),
         (Series(['b', 'a', 'c', 'a', 'c']),
          ['a', 'b', 'c'],
          Series([1, 0, 2, 0, 2], dtype=np.int32),
          Series(['1', '2', '3', '4']))])
def test_fit_transform(values, keys, expected_encoded, unknown):
    # Test LabelEncoder's fit_transform and it's reaction to unkown label
    le = LabelEncoder()
    encoded = le.fit_transform(values)
    assert(len(encoded) == len(expected_encoded))
    assert(len(encoded) == len(encoded[encoded == expected_encoded]))

    assert(le._fitted is True)

    with pytest.raises(ValueError, match='contains previously unseen labels'):
        le.transform(unknown)


def test_transform_without_fit():
    # Test if ValueError is raised if it is asked to transform without fit
    values = dc.from_cudf(Series(['b', 'a', 'c', 'a']), npartitions=2)
    le = LabelEncoder()
    assert(le._fitted is False)
    with pytest.raises(ValueError, match='LabelEncoder must be fit first'):
        le.transform(values)


@pytest.mark.parametrize(
        "values",
        [[1, 2, 3],
            DataFrame({'a': range(10)}),
            np.array(['a', 'b', 'c']),
            None])
def test_bad_input_type(values):
    # Test fit with bad input_type
    le = LabelEncoder()
    with pytest.raises(TypeError,
                       match='not dask_cudf.Series or cudf.Series'):
        le.fit(values)

    # Test fit_transform with bad input_type
    le = LabelEncoder()
    with pytest.raises(TypeError,
                       match='not dask_cudf.Series or cudf.Series'):
        le.fit_transform(values)


@pytest.mark.parametrize(
        "orig_label, ord_label, expected_reverted, bad_ord_label",
        [(dc.from_cudf(Series(['a', 'b', 'c']), npartitions=2),
          dc.from_cudf(Series([2, 1, 2, 0]), npartitions=2),
          dc.from_cudf(Series(['c', 'b', 'c', 'a']), npartitions=2),
          dc.from_cudf(Series([-1, 1, 2, 0]), npartitions=2)),
         (dc.from_cudf(Series(['Tokyo', 'Paris', 'Austin']), npartitions=2),
          dc.from_cudf(Series([0, 2, 0]), npartitions=2),
          dc.from_cudf(Series(['Austin', 'Tokyo', 'Austin']), npartitions=2),
          dc.from_cudf(Series([0, 1, 2, 3]), npartitions=2))])
def test_inverse_transform(orig_label, ord_label,
                           expected_reverted, bad_ord_label):
    # prepare LabelEncoder
    le = LabelEncoder()
    le.fit(orig_label)
    assert(le._fitted is True)

    # test if inverse_transform is correct
    reverted = le.inverse_transform(ord_label)
    assert(len(reverted) == len(expected_reverted))
    assert(len(reverted)
           == len(reverted[reverted == expected_reverted]))

    # test if correctly raises ValueError
    with pytest.raises(ValueError):
        reverted = le.inverse_transform(bad_ord_label)
        reverted = reverted.compute()


def test_unfitted_inverse_transform():
    """ Try calling `.inverse_transform()` without fitting first
    """
    orig_label = Series(np.random.choice(10, (10,)))
    le = LabelEncoder()
    assert(not le._fitted)

    with pytest.raises(ValueError, match='LabelEncoder must be fit first'):
        le.transform(orig_label)
