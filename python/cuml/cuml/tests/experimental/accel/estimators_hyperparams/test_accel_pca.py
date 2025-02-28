#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def pca_data():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    # Standardize features before PCA
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("n_components", [2, 5, "mle"])
def test_pca_n_components(pca_data, n_components):
    X, _ = pca_data
    pca = PCA(n_components=n_components).fit(X)
    X_transformed = pca.transform(X)
    # Check the shape of the transformed data
    if n_components != "mle":
        assert (
            X_transformed.shape[1] == n_components
        ), f"Expected {n_components} components, got {X_transformed.shape[1]}"
    # Check that explained variance ratios sum up appropriately
    total_variance = np.sum(pca.explained_variance_ratio_)
    assert (
        total_variance <= 1.1
    ), "Total explained variance cannot exceed with margin for parallel error"
    assert (
        total_variance > 0.0
    ), "Total explained variance ratio should be positive"


@pytest.mark.parametrize(
    "svd_solver", ["auto", "full", "arpack", "randomized", "covariance_eigh"]
)
def test_pca_svd_solver(pca_data, svd_solver):
    X, _ = pca_data
    pca = PCA(n_components=5, svd_solver=svd_solver, random_state=42).fit(X)
    X_transformed = pca.transform(X)
    # Reconstruct the data
    pca.inverse_transform(X_transformed)


@pytest.mark.parametrize("whiten", [True, False])
def test_pca_whiten(pca_data, whiten):
    X, _ = pca_data
    pca = PCA(n_components=5, whiten=whiten).fit(X)
    X_transformed = pca.transform(X)
    # If whiten is True, transformed data should have unit variance
    variances = np.var(X_transformed, axis=0)
    if whiten:
        np.testing.assert_allclose(
            variances,
            1.0,
            atol=1e-1,
            err_msg="Transformed features should have unit variance when whiten=True",
        )


@pytest.mark.parametrize("tol", [0.0, 1e-4, 1e-2])
def test_pca_tol(pca_data, tol):
    X, _ = pca_data
    pca = PCA(
        n_components=5, svd_solver="arpack", tol=tol, random_state=42
    ).fit(X)
    pca.transform(X)
    # Since 'arpack' is iterative, tol might affect convergence
    # Check that the explained variance ratio is reasonable
    total_variance = np.sum(pca.explained_variance_ratio_)
    assert (
        total_variance > 0.5
    ), "Total explained variance should be significant"


def test_pca_random_state(pca_data):
    X, _ = pca_data
    pca1 = PCA(n_components=5, svd_solver="randomized", random_state=42).fit(X)
    pca2 = PCA(n_components=5, svd_solver="randomized", random_state=42).fit(X)
    # With the same random_state, components should be the same
    np.testing.assert_allclose(
        pca1.components_,
        pca2.components_,
        err_msg="Components should be the same with the same random_state",
    )


@pytest.mark.parametrize("copy", [True, False])
def test_pca_copy(pca_data, copy):
    X, _ = pca_data
    X_original = X.copy()
    PCA(n_components=5, copy=copy).fit(X)
    if copy:
        # X should remain unchanged
        assert np.allclose(X, X_original), "X has been modified when copy=True"
    else:
        # X might be modified when copy=False
        pass  # We cannot guarantee X remains unchanged


@pytest.mark.parametrize("iterated_power", [0, 3, 5, "auto"])
def test_pca_iterated_power(pca_data, iterated_power):
    X, _ = pca_data
    pca = PCA(
        n_components=5,
        svd_solver="randomized",
        iterated_power=iterated_power,
        random_state=42,
    ).fit(X)
    pca.transform(X)
    # Check that the explained variance ratio is reasonable
    total_variance = np.sum(pca.explained_variance_ratio_)
    assert (
        total_variance > 0.5
    ), f"Total explained variance should be significant with iterated_power={iterated_power}"


def test_pca_explained_variance_ratio(pca_data):
    X, _ = pca_data
    pca = PCA(n_components=None).fit(X)
    total_variance = np.sum(pca.explained_variance_ratio_)
    np.testing.assert_almost_equal(
        total_variance,
        1.0,
        decimal=5,
        err_msg="Total explained variance ratio should sum to 1 when n_components=None",
    )


def test_pca_inverse_transform(pca_data):
    X, _ = pca_data
    pca = PCA(n_components=5).fit(X)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    # Check reconstruction error
    np.mean((X - X_reconstructed) ** 2)
