# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from sklearn.datasets import load_breast_cancer, load_wine

from cuml import LinearRegression

X_cancer, y_cancer = load_breast_cancer(return_X_y=True, as_frame=True)
X_wine, y_wine = load_wine(return_X_y=True, as_frame=True)
X_np, y_np = load_wine(return_X_y=True, as_frame=False)  # No feature names


def test_no_feature_names_in_attribute():
    lr = LinearRegression().fit(X_np, y_np)
    assert not hasattr(lr, "feature_names_in_")


def test_feature_names_in_attribute():
    lr = LinearRegression().fit(X_cancer, y_cancer)
    assert len(lr.feature_names_in_) == 30

    lr.fit(X_wine, y_wine)
    assert len(lr.feature_names_in_) == 13

    lr.fit(X_np, y_np)
    assert not hasattr(lr, "feature_names_in_")


def test_feature_names_in_mixed_types():
    X, y = X_cancer.copy(), y_cancer.copy()
    X.columns = [0] + [str(i) for i in range(1, len(X.columns))]

    with pytest.raises(
        TypeError, match=".*only supported if all input features.*"
    ):
        LinearRegression().fit(X, y)


def test_feature_names_in_not_in_fit():
    lr = LinearRegression().fit(X_cancer.to_numpy(), y_cancer)
    with pytest.warns(UserWarning, match=".*X has feature names.*"):
        lr.predict(X_cancer)


def test_feature_names_in_not_in_predict():
    lr = LinearRegression().fit(X_cancer, y_cancer)
    with pytest.warns(
        UserWarning, match=".*X does not have valid feature names.*"
    ):
        lr.predict(X_cancer.to_numpy())


def test_invalid_feature_names_in_():
    X = X_cancer.copy()
    X.columns = [str(i) for i in range(len(X.columns))]

    lr = LinearRegression().fit(X_cancer, y_cancer)
    with pytest.raises(ValueError, match=".*The feature names should match.*"):
        lr.predict(X)


def test_n_features_in_attribute():
    lr = LinearRegression().fit(X_np, y_np)
    assert lr.n_features_in_ == X_np.shape[1]

    lr = lr.fit(X_cancer, y_cancer)
    assert lr.n_features_in_ == X_cancer.shape[1]
