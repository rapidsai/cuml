# Copyright (c) 2025, NVIDIA CORPORATION.
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
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=2,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="module")
def multiclass():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


def test_svc_linear(binary):
    X, y = binary
    svc = LinearSVC().fit(X, y)
    assert svc.score(X, y) > 0.5


def test_svc_linear_multiclass(multiclass):
    X, y = multiclass
    svr = LinearSVC().fit(X, y)
    assert svr.score(X, y) > 0.5
