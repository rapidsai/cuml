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
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


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


def test_svr_linear(linear_X_y):
    X, y = linear_X_y
    svr = LinearSVR().fit(X, y)
    assert svr.score(X, y) > 0.5
