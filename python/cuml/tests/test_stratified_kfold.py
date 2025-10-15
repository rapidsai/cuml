# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import numpy as np
import pytest

from cuml.datasets import make_regression
from cuml.model_selection import KFold, StratifiedKFold


def get_x_y(n_samples, n_classes):
    X = cudf.DataFrame({"x": range(n_samples)})
    y = cp.arange(n_samples) % n_classes
    cp.random.shuffle(y)
    y = cudf.Series(y)
    return X, y


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_splits", [5, 10])
@pytest.mark.parametrize("n_samples", [10000])
@pytest.mark.parametrize("n_classes", [2, 10])
def test_split_dataframe(n_samples, n_classes, n_splits, shuffle):
    X, y = get_x_y(n_samples, n_classes)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    assert kf.get_n_splits(X, y) == kf.get_n_splits() == n_splits

    for train_index, test_index in kf.split(X, y):
        assert len(train_index) + len(test_index) == n_samples
        assert len(train_index) == len(test_index) * (n_splits - 1)
        for i in range(n_classes):
            ratio_tr = (y[train_index] == i).sum() / len(train_index)
            ratio_te = (y[test_index] == i).sum() / len(test_index)
            assert ratio_tr == ratio_te


def test_num_classes_check():
    X, y = get_x_y(n_samples=1000, n_classes=1)
    kf = StratifiedKFold(n_splits=5)
    err_msg = "number of unique classes cannot be less than 2"
    with pytest.raises(ValueError, match=err_msg):
        for train_index, test_index in kf.split(X, y):
            pass


@pytest.mark.parametrize("n_splits", [0, 1])
def test_invalid_folds(n_splits):
    X, y = get_x_y(n_samples=1000, n_classes=2)

    err_msg = f"n_splits {n_splits} is not a integer at least 2"
    with pytest.raises(ValueError, match=err_msg):
        kf = StratifiedKFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X, y):
            break


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_splits", [5, 10])
@pytest.mark.parametrize(
    "random_state",
    [
        1,
        np.random.RandomState(1),
        cp.random.RandomState(1),
        None,
    ],
)
def test_kfold(shuffle, n_splits, random_state) -> None:
    n_samples = 256
    n_features = 16
    X, y = make_regression(n_samples, n_features, random_state=1)
    kfold = KFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    assert kfold.get_n_splits(X, y) == kfold.get_n_splits() == n_splits
    n_test_total = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        n_test_total += test_idx.size

        assert train_idx.shape[0] + test_idx.shape[0] == n_samples
        fold_size = X.shape[0] // n_splits
        # We assign the remainder to the beginning folds.
        if fold_idx < n_samples % n_splits:
            assert test_idx.shape[0] == fold_size + 1
        else:
            assert test_idx.shape[0] == fold_size
        assert cp.all(train_idx >= 0)
        assert cp.all(test_idx >= 0)
        indices = cp.concatenate([train_idx, test_idx])
        assert len(indices.shape) == 1
        assert indices.size == n_samples
        uniques = cp.unique(indices)
        sorted_uniques = cp.sort(uniques)

        assert uniques.size == n_samples, indices
        arr = cp.arange(n_samples)
        cp.testing.assert_allclose(sorted_uniques, arr)

    assert n_test_total == n_samples


# Since the kfold only uses the shape of the input, not the actual data, we only have a
# small test for dataframe.
def test_kfold_dataframe() -> None:
    n_samples = 4096
    X, y = get_x_y(n_samples, 2)
    kfold = KFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in kfold.split(X, y):
        assert train_idx.shape[0] + test_idx.shape[0] == n_samples
