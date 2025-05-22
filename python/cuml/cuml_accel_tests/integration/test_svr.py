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

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@pytest.fixture(scope="module")
def linear_X_y():
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.fixture(scope="module")
def sinusoid_X_y():
    rng = np.random.default_rng(42)
    N = 200
    X = np.sort(5 * rng.random((N, 1)), axis=0)
    y = np.sin(X).ravel()

    # add noise to targets
    y[::5] += 3 * (0.5 - rng.random(N // 5))
    return X, y


def test_svr_linear(linear_X_y):
    X, y = linear_X_y
    svr = SVR(kernel="linear").fit(X, y)
    assert svr.score(X, y) > 0.5


@pytest.mark.parametrize("kernel", ["poly", "rbf"])
def test_svr_sinusoid(sinusoid_X_y, kernel):
    X, y = sinusoid_X_y
    svr = SVR(kernel=kernel).fit(X, y)
    assert svr.score(X, y) > 0.5
