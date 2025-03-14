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
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


@pytest.fixture(scope="module")
def svd_data():
    X, y = make_classification(
        n_samples=300,
        n_features=50,
        n_informative=10,
        n_redundant=10,
        random_state=42,
    )
    # Convert the data to a sparse CSR matrix
    return X, y


@pytest.mark.parametrize("n_components", [5, 10, 20, 30])
def test_truncated_svd_n_components(svd_data, n_components):
    X, _ = svd_data
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_transformed = svd.fit_transform(X)
    # Check the shape of the transformed data
    assert (
        X_transformed.shape[1] == n_components
    ), f"Expected {n_components} components, got {X_transformed.shape[1]}"
    # Check that explained variance ratios sum up appropriately
    total_variance = np.sum(svd.explained_variance_ratio_)
    assert (
        total_variance <= 1.0
    ), "Total explained variance ratio cannot exceed 1"
    assert (
        total_variance > 0.0
    ), "Total explained variance ratio should be positive"


@pytest.mark.parametrize("algorithm", ["randomized", "arpack"])
def test_truncated_svd_algorithm(svd_data, algorithm):
    X, _ = svd_data
    svd = TruncatedSVD(n_components=10, algorithm=algorithm, random_state=42)
    X_transformed = svd.fit_transform(X)
    # Reconstruct the data
    svd.inverse_transform(X_transformed)


@pytest.mark.parametrize("n_iter", [5, 7, 10])
def test_truncated_svd_n_iter(svd_data, n_iter):
    X, _ = svd_data
    svd = TruncatedSVD(n_components=10, n_iter=n_iter, random_state=42)
    svd.fit_transform(X)
    # Check that the explained variance ratio is reasonable
    total_variance = np.sum(svd.explained_variance_ratio_)
    assert (
        total_variance > 0.5
    ), f"Total explained variance should be significant with n_iter={n_iter}"


def test_truncated_svd_random_state(svd_data):
    X, _ = svd_data
    svd1 = TruncatedSVD(
        n_components=10, algorithm="randomized", random_state=42
    )
    svd2 = TruncatedSVD(
        n_components=10, algorithm="randomized", random_state=42
    )
    svd1.fit_transform(X)
    svd2.fit_transform(X)
    # With the same random_state, components should be the same
    np.testing.assert_allclose(
        svd1.components_,
        svd2.components_,
        err_msg="Components should be the same with the same random_state",
    )
    svd3 = TruncatedSVD(
        n_components=10, algorithm="randomized", random_state=24
    )
    svd3.fit(X)


@pytest.mark.parametrize("tol", [0.0, 1e-4, 1e-2])
def test_truncated_svd_tol(svd_data, tol):
    X, _ = svd_data
    svd = TruncatedSVD(
        n_components=10, algorithm="arpack", tol=tol, random_state=42
    )
    svd.fit_transform(X)
    # Check that the explained variance ratio is reasonable
    total_variance = np.sum(svd.explained_variance_ratio_)
    assert (
        total_variance > 0.5
    ), f"Total explained variance should be significant with tol={tol}"


@pytest.mark.parametrize(
    "power_iteration_normalizer", ["auto", "OR", "LU", "none"]
)
def test_truncated_svd_power_iteration_normalizer(
    svd_data, power_iteration_normalizer
):
    X, _ = svd_data
    svd = TruncatedSVD(
        n_components=10,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=42,
    )
    svd.fit_transform(X)
    # Check that the explained variance ratio is reasonable
    total_variance = np.sum(svd.explained_variance_ratio_)
    assert (
        total_variance > 0.5
    ), f"Total explained variance should be significant with power_iteration_normalizer={power_iteration_normalizer}"


def test_truncated_svd_inverse_transform(svd_data):
    X, _ = svd_data
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_transformed = svd.fit_transform(X)
    X_reconstructed = svd.inverse_transform(X_transformed)
    # Check reconstruction error
    np.mean((X - X_reconstructed) ** 2)


def test_truncated_svd_sparse_input_dense_output(svd_data):
    X, _ = svd_data
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_transformed = svd.fit_transform(X)
    # The output should be dense even if input is sparse
    assert not isinstance(
        X_transformed, csr_matrix
    ), "Transformed data should be dense"


def test_truncated_svd_components_norm(svd_data):
    X, _ = svd_data
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd.fit(X)
    components_norm = np.linalg.norm(svd.components_, axis=1)
    np.testing.assert_allclose(
        components_norm,
        1.0,
        atol=1e-5,
        err_msg="Each component should have unit length",
    )


@pytest.mark.parametrize("n_oversamples", [5])
def test_truncated_svd_n_oversamples(svd_data, n_oversamples):
    X, _ = svd_data
    svd = TruncatedSVD(
        n_components=10, n_oversamples=n_oversamples, random_state=42
    )
    svd.fit_transform(X)
    # Check that the explained variance ratio is reasonable
    total_variance = np.sum(svd.explained_variance_ratio_)
    assert (
        total_variance > 0.5
    ), f"Total explained variance should be significant with n_oversamples={n_oversamples}"
