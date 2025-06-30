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

import cupy as cp
import cupyx.scipy.sparse
import numpy as np
import pytest
import scipy.sparse
from sklearn import datasets
from sklearn.manifold import SpectralEmbedding as skSpectralEmbedding
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.extmath import _deterministic_vector_sign_flip

# from sklearn.cluster import KMeans
from cuml.cluster import KMeans
from cuml.manifold import SpectralEmbedding
from cuml.metrics import trustworthiness as cuml_trustworthiness
from cuml.testing.utils import (
    array_equal,
    quality_param,
    stress_param,
    unit_param,
)

# Default testing parameters
DEFAULT_N_NEIGHBORS = 12
DEFAULT_N_COMPONENTS = 2


def validate_embedding(X, Y, score=0.70, n_neighbors=DEFAULT_N_NEIGHBORS):
    """Compares Spectral Embedding trustworthiness and NANs"""
    if isinstance(Y, cp.ndarray):
        nans = cp.sum(cp.isnan(Y))
        print(X.shape, Y.shape)
        trust = cuml_trustworthiness(X, Y, n_neighbors=n_neighbors)
    else:
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X, Y, n_neighbors=n_neighbors)

    print("Trust=%s" % trust)
    assert trust > score
    assert nans == 0


@pytest.mark.parametrize("n_components", [2, 5, 10])
def test_spectral_embedding_components(
    supervised_learning_dataset, n_components
):
    """Test that spectral embedding respects n_components parameter"""
    X = supervised_learning_dataset

    spectral = SpectralEmbedding(n_components=n_components, random_state=42)
    embedding = spectral.fit_transform(X)

    assert embedding.shape[1] == n_components


# @pytest.mark.parametrize("n_components", [2, 5])
# @pytest.mark.parametrize("norm_laplacian", [True, False])
# @pytest.mark.parametrize("drop_first", [True, False])
# def test_spectral_embedding_params(
#     supervised_learning_dataset, n_components, norm_laplacian, drop_first
# ):
#     """Test various parameter combinations of spectral embedding"""
#     X = supervised_learning_dataset

#     spectral = SpectralEmbedding(
#         n_components=n_components,
#         random_state=42,
#         n_neighbors=DEFAULT_N_NEIGHBORS
#     )

#     embedding = spectral._fit(
#         X, n_components,
#         random_state=42,
#         n_neighbors=DEFAULT_N_NEIGHBORS,
#         norm_laplacian=norm_laplacian,
#         drop_first=drop_first
#     )

#     validate_embedding(X, embedding)

# @pytest.mark.parametrize("input_type", ["cupy", "scipy"])
# def test_spectral_embedding_sparse_input(input_type):
#     """Test spectral embedding with sparse inputs"""
#     # Create a sparse dataset
#     X, y = datasets.make_blobs(
#         n_samples=500, n_features=20, centers=5, random_state=42
#     )

#     # Sparsify the data (set 50% of entries to zero)
#     sparsification = np.random.choice(
#         [0.0, 1.0], p=[0.5, 0.5], size=X.shape
#     )
#     X_sparse = np.multiply(X, sparsification)

#     if input_type == "cupy":
#         X_sparse = cupyx.scipy.sparse.csr_matrix(X_sparse)
#     else:
#         X_sparse = scipy.sparse.csr_matrix(X_sparse)

#     spectral = SpectralEmbedding(
#         n_components=2,
#         random_state=42,
#         n_neighbors=DEFAULT_N_NEIGHBORS
#     )

#     embedding = spectral.fit_transform(X_sparse)

#     validate_embedding(X, embedding)


@pytest.mark.parametrize("random_state", [None, 42, 999])
def test_spectral_embedding_reproducibility(
    supervised_learning_dataset, random_state
):
    """Test that spectral embedding is reproducible with fixed random state"""
    X = supervised_learning_dataset

    # First embedding
    spectral1 = SpectralEmbedding(
        n_components=DEFAULT_N_COMPONENTS,
        random_state=random_state,
        n_neighbors=DEFAULT_N_NEIGHBORS,
    )
    embedding1 = spectral1.fit_transform(X)

    # Second embedding
    spectral2 = SpectralEmbedding(
        n_components=DEFAULT_N_COMPONENTS,
        random_state=random_state,
        n_neighbors=DEFAULT_N_NEIGHBORS,
    )
    embedding2 = spectral2.fit_transform(X)

    # If random state is None, embeddings may differ
    if random_state is not None:
        assert array_equal(embedding1, embedding2, 1e-4, with_sign=True)


@pytest.mark.parametrize("n_neighbors", [5, 10, 15])
def test_spectral_embedding_n_neighbors(
    supervised_learning_dataset, n_neighbors
):
    """Test different number of neighbors impact on spectral embedding"""
    X = supervised_learning_dataset

    spectral = SpectralEmbedding(
        n_components=DEFAULT_N_COMPONENTS,
        random_state=42,
        n_neighbors=n_neighbors,
    )
    embedding = spectral.fit_transform(X)

    validate_embedding(X, embedding, n_neighbors=n_neighbors)


@pytest.mark.parametrize("dataset_name", ["blobs", "iris", "digits"])
def test_spectral_embedding_datasets(dataset_name):
    """Test spectral embedding on various datasets"""
    if dataset_name == "blobs":
        X, y = datasets.make_blobs(
            n_samples=300, n_features=10, centers=5, random_state=42
        )
    elif dataset_name == "iris":
        X, y = datasets.load_iris(return_X_y=True)
    elif dataset_name == "digits":
        X, y = datasets.load_digits(return_X_y=True)

    spectral = SpectralEmbedding(
        n_components=2, random_state=42, n_neighbors=DEFAULT_N_NEIGHBORS
    )
    embedding = spectral.fit_transform(X)

    print(X.shape, embedding.shape)
    validate_embedding(X, embedding, n_neighbors=DEFAULT_N_NEIGHBORS)


@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(2000), stress_param(10000)]
)
@pytest.mark.parametrize(
    "n_feats", [unit_param(10), quality_param(50), stress_param(100)]
)
def test_spectral_embedding_scaling(nrows, n_feats):
    """Test spectral embedding on larger datasets with varying dimensions"""
    X, y = datasets.make_blobs(
        n_samples=nrows, n_features=n_feats, centers=5, random_state=42
    )

    spectral = SpectralEmbedding(
        n_components=2, random_state=42, n_neighbors=DEFAULT_N_NEIGHBORS
    )

    embedding = spectral.fit_transform(X)

    if nrows <= 2000:  # Only check trustworthiness for smaller datasets
        validate_embedding(X, embedding)


def test_compare_sklearn():
    """Compare results with sklearn's implementation for basic correctness check"""
    X, y = datasets.load_digits(return_X_y=True)

    # cuML implementation
    cuml_spectral = SpectralEmbedding(
        n_components=2, random_state=42, n_neighbors=DEFAULT_N_NEIGHBORS
    )
    cuml_embedding = cuml_spectral.fit_transform(X)

    # sklearn implementation
    sk_spectral = skSpectralEmbedding(
        n_components=2, random_state=42, n_neighbors=DEFAULT_N_NEIGHBORS
    )
    sk_embedding = sk_spectral.fit_transform(X)

    cuml_embedding = cuml_embedding.get()
    cuml_embedding = cuml_embedding.T
    cuml_embedding = _deterministic_vector_sign_flip(cuml_embedding)
    cuml_embedding = cuml_embedding.T

    print(cuml_embedding.shape, sk_embedding.shape)
    print(cuml_embedding)
    print(sk_embedding)

    # check if cuml_embedding is close to sk_embedding
    assert array_equal(
        cuml_embedding, sk_embedding, unit_tol=1e-2, with_sign=True
    )

    print("cuml trust")
    validate_embedding(X, cuml_embedding)
    print("sklearn trust")
    validate_embedding(X, sk_embedding)

    # # Check that both embeddings have similar clustering quality
    # cuml_score = adjusted_rand_score(y, KMeans(5).fit_predict(cuml_embedding))
    # sk_score = adjusted_rand_score(y, KMeans(5).fit_predict(sk_embedding))

    # # The scores don't need to be identical, but should be comparable
    # assert abs(cuml_score - sk_score) < 0.2
