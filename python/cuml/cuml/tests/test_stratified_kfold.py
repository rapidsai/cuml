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
import pytest

from cuml.model_selection import StratifiedKFold


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
