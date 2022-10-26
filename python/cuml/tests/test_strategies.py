# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cuml.testing.strategies import (
    datasets,
    regression_datasets,
    split_datasets,
)
from hypothesis import given


@given(datasets())
def test_datasets(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert y.ndim in (0, 1, 2)


@given(split_datasets())
def test_split_datasets(split_dataset):
    X_train, X_test, y_train, y_test = split_dataset

    assert X_train.ndim == X_test.ndim == 2
    assert y_train.ndim == y_test.ndim
    assert y_train.ndim in (0, 1, 2)


@given(regression_datasets())
def test_regression_datasets(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert y.ndim in (0, 1, 2)


@given(split_datasets(datasets=regression_datasets()))
def test_split_regression_datasets(split_dataset):
    X_train, X_test, y_train, y_test = split_dataset

    assert X_train.ndim == X_test.ndim == 2
    assert y_train.ndim == y_test.ndim
    assert y_train.ndim in (0, 1, 2)
