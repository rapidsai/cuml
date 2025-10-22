# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
    load_digits,
    make_circles,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.manifold import SpectralEmbedding as skSpectralEmbedding
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from cuml.manifold import SpectralEmbedding, spectral_embedding
from cuml.manifold.umap import fuzzy_simplicial_set
from cuml.testing.datasets import make_classification_dataset

# Test parameters
N_NEIGHBORS = 15
N_COMPONENTS = 2


def generate_s_curve(n_samples):
    """Generate S-curve dataset."""
    X, _ = make_s_curve(n_samples=n_samples, noise=0.05, random_state=42)
    return X


def generate_swiss_roll(n_samples):
    """Generate Swiss Roll dataset."""
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    return X


def generate_mnist_like_dataset(n_samples):
    """Load and sample dataset using cuML's testing infrastructure."""

    # Generate a classification dataset with similar characteristics to MNIST
    # MNIST has 784 features (28x28 pixels) and 10 classes
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=np.float32,
        nrows=n_samples,
        ncols=784,  # Same as MNIST features
        n_info=100,  # Number of informative features
        num_classes=10,  # Same as MNIST classes
    )

    # Normalize to [0, 1] range like MNIST
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    return X_train


def load_digits_dataset(n_samples=None):
    """Load digits dataset (n_samples is ignored as dataset has fixed size)."""
    digits = load_digits()
    return digits.data


@pytest.mark.parametrize(
    "affinity,graph_type",
    [
        ("nearest_neighbors", None),  # Use built-in nearest_neighbors affinity
        ("precomputed", "binary_knn"),  # Precomputed binary k-NN graph
        (
            "precomputed",
            "distance_knn",
        ),  # Precomputed k-NN graph with distances
        ("precomputed", "fuzzy_knn"),  # Precomputed fuzzy k-NN graph from UMAP
    ],
)
@pytest.mark.parametrize(
    "dataset_loader,n_samples",
    [
        (generate_s_curve, 1500),
        (generate_s_curve, 2000),
        (generate_swiss_roll, 2000),
        (generate_swiss_roll, 3000),
        (generate_mnist_like_dataset, 5000),
        (load_digits_dataset, None),
    ],
)
def test_spectral_embedding_trustworthiness(
    dataset_loader, n_samples, affinity, graph_type
):
    """Test trustworthiness comparison between sklearn and cuML on various datasets.

    Tests different graph construction methods:
    - nearest_neighbors affinity: Uses built-in k-NN graph construction
    - precomputed with binary_knn: Binary connectivity k-NN graph
    - precomputed with distance_knn: k-NN graph with distance weights
    - precomputed with fuzzy_knn: Smooth weighted graph from UMAP's fuzzy simplicial set
    """
    # Load/generate dataset
    X = dataset_loader(n_samples) if n_samples else dataset_loader(None)

    if affinity == "precomputed":
        if graph_type == "fuzzy_knn":
            # Use fuzzy_simplicial_set to create a smooth weighted KNN graph
            X_gpu = cp.asarray(X, dtype=np.float32)

            # Create smooth KNN graph using fuzzy_simplicial_set
            # This creates a weighted graph with fuzzy membership strengths
            graph = fuzzy_simplicial_set(
                X_gpu,
                n_neighbors=N_NEIGHBORS,
                random_state=42,
            )

            # sklearn embedding with precomputed fuzzy graph
            sk_spectral = skSpectralEmbedding(
                n_components=N_COMPONENTS,
                affinity="precomputed",
                random_state=42,
            )
            X_sklearn = sk_spectral.fit_transform(graph.get())

            # cuML embedding with precomputed fuzzy graph
            cuml_spectral = SpectralEmbedding(
                n_components=N_COMPONENTS,
                affinity="precomputed",
                random_state=42,
            )
            X_cuml_gpu = cuml_spectral.fit_transform(graph)
            X_cuml = cp.asnumpy(X_cuml_gpu)

        elif graph_type in ["binary_knn", "distance_knn"]:
            # Create k-neighbors graph for precomputed affinity
            mode = "connectivity" if graph_type == "binary_knn" else "distance"
            knn_graph = kneighbors_graph(
                X,
                n_neighbors=N_NEIGHBORS,
                mode=mode,
                include_self=True,
            )
            # Make symmetric
            knn_graph = 0.5 * (knn_graph + knn_graph.T)
            knn_coo = knn_graph.tocoo()

            # sklearn embedding with precomputed
            sk_spectral = skSpectralEmbedding(
                n_components=N_COMPONENTS,
                affinity="precomputed",
                random_state=42,
            )
            X_sklearn = sk_spectral.fit_transform(knn_coo)

            # cuML embedding with precomputed
            cuml_spectral = SpectralEmbedding(
                n_components=N_COMPONENTS,
                affinity="precomputed",
                random_state=42,
            )
            X_cuml_gpu = cuml_spectral.fit_transform(knn_coo)
            X_cuml = cp.asnumpy(X_cuml_gpu)
    else:
        # sklearn embedding with nearest_neighbors
        sk_spectral = skSpectralEmbedding(
            n_components=N_COMPONENTS,
            n_neighbors=N_NEIGHBORS,
            affinity="nearest_neighbors",
            random_state=42,
            n_jobs=-1,
        )
        X_sklearn = sk_spectral.fit_transform(X)

        # cuML embedding with nearest_neighbors
        X_gpu = cp.asarray(X)
        cuml_spectral = SpectralEmbedding(
            n_components=N_COMPONENTS,
            affinity="nearest_neighbors",
            n_neighbors=N_NEIGHBORS,
            random_state=42,
        )
        X_cuml_gpu = cuml_spectral.fit_transform(X_gpu)
        X_cuml = cp.asnumpy(X_cuml_gpu)

    # Calculate trustworthiness scores
    trust_sklearn = trustworthiness(X, X_sklearn, n_neighbors=N_NEIGHBORS)
    trust_cuml = trustworthiness(X, X_cuml, n_neighbors=N_NEIGHBORS)

    # Assertions
    min_trustworthiness = 0.8
    assert trust_sklearn > min_trustworthiness
    assert trust_cuml > min_trustworthiness


def test_spectral_embedding_function_api():
    """Smoke test for spectral_embedding function: reproducibility and output shape."""
    # Generate S-curve dataset
    n_samples = 500
    X, _ = make_s_curve(n_samples=n_samples, noise=0.05, random_state=42)
    X = X.astype(np.float32)
    X_gpu = cp.asarray(X)

    # Test 1: Output shape validation
    for n_components in [1, 2, 3]:
        embedding = spectral_embedding(
            X_gpu, n_components=n_components, random_state=42
        )
        assert embedding.shape == (
            n_samples,
            n_components,
        )

    # Test 2: Reproducibility with seed
    seed = 123
    embedding1 = spectral_embedding(X_gpu, n_components=2, random_state=seed)
    embedding2 = spectral_embedding(X_gpu, n_components=2, random_state=seed)

    assert cp.allclose(embedding1, embedding2)


def test_spectral_embedding_invalid_affinity():
    X, _ = make_s_curve(n_samples=200, noise=0.05, random_state=42)
    with pytest.raises(
        ValueError, match="`affinity='oops!'` is not supported"
    ):
        spectral_embedding(X, affinity="oops!")


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
@pytest.mark.parametrize("affinity", ["nearest_neighbors", "precomputed"])
def test_spectral_embedding_nonfinite(value, affinity):
    X = np.array([[0, 1], [2, 3], [0, value]], dtype="float32")

    with pytest.raises(ValueError, match="nonfinite"):
        spectral_embedding(X, affinity=affinity)


@pytest.mark.parametrize(
    "input_type,expected_type",
    [
        ("numpy", np.ndarray),
        ("cupy", cp.ndarray),
    ],
)
def test_output_type_handling(input_type, expected_type):
    """Test that output types are properly handled for different input types."""
    # Generate test data
    n_samples = 500
    X_np, _ = make_s_curve(n_samples=n_samples, noise=0.05, random_state=42)
    X_np = X_np.astype(np.float32)

    # Convert to appropriate type
    X = X_np if input_type == "numpy" else cp.asarray(X_np)

    # Test spectral_embedding function
    embedding = spectral_embedding(X, n_components=2, random_state=42)
    assert isinstance(embedding, expected_type)
    assert embedding.shape == (n_samples, 2)

    # Test SpectralEmbedding class with fit
    model = SpectralEmbedding(n_components=2, n_neighbors=15, random_state=42)
    model.fit(X)
    assert isinstance(model.embedding_, expected_type)
    assert model.embedding_.shape == (n_samples, 2)

    # Test fit_transform
    out = SpectralEmbedding(
        n_components=2, n_neighbors=15, random_state=42
    ).fit_transform(X)
    assert isinstance(out, expected_type)
    assert out.shape == (n_samples, 2)


@pytest.mark.parametrize(
    "converter",
    [
        pytest.param(lambda x: x.toarray(), id="numpy"),
        pytest.param(lambda x: cp.asarray(x.toarray()), id="cupy"),
        pytest.param(sp.coo_matrix, id="scipy_coo"),
        pytest.param(sp.csr_matrix, id="scipy_csr"),
        pytest.param(sp.csc_matrix, id="scipy_csc"),
        pytest.param(cp_sp.coo_matrix, id="cupy_coo"),
        pytest.param(cp_sp.csr_matrix, id="cupy_csr"),
        pytest.param(cp_sp.csc_matrix, id="cupy_csc"),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_precomputed_matrix_formats(converter, dtype):
    """Test that various matrix formats work correctly with precomputed affinity.

    This test verifies that SpectralEmbedding works with all combinations of:
    - Matrix formats: COO, CSR, CSC, and dense
    - Libraries: scipy and cupy
    - dtypes: float32 and float64

    It also ensures the embeddings have good trustworthiness scores.
    """

    # Generate test data using existing helper function
    n_samples = 1000
    X_np = generate_s_curve(n_samples)

    # Create a symmetric k-NN affinity graph
    knn_graph = kneighbors_graph(
        X_np,
        n_neighbors=N_NEIGHBORS,
        mode="connectivity",
        include_self=True,
    )
    knn_graph = 0.5 * (knn_graph + knn_graph.T)

    # Convert to the desired format
    affinity_matrix = converter(knn_graph).astype(dtype)

    # Test with SpectralEmbedding class
    model = SpectralEmbedding(
        n_components=2, affinity="precomputed", random_state=42
    )
    embedding_class = model.fit_transform(affinity_matrix)

    # Test with spectral_embedding function
    embedding_func = spectral_embedding(
        affinity_matrix,
        n_components=2,
        affinity="precomputed",
        random_state=42,
    )

    # Verify output shapes
    assert embedding_class.shape == (n_samples, 2)
    assert embedding_func.shape == (n_samples, 2)

    # Calculate and print trustworthiness scores
    trust_class = trustworthiness(
        X_np, cp.asnumpy(embedding_class), n_neighbors=N_NEIGHBORS
    )
    trust_func = trustworthiness(
        X_np, cp.asnumpy(embedding_func), n_neighbors=N_NEIGHBORS
    )

    # Verify embeddings have good quality
    min_trust = 0.8
    assert trust_class > min_trust
    assert trust_func > min_trust


def test_precomputed_no_sparsity():
    """This test ensures that embedding works in the rare case where affinity is 0% sparse."""
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
    distances = pairwise_distances(X)
    gamma = 1.0
    affinity_matrix = np.exp(-gamma * distances**2)
    affinity_matrix = sp.coo_matrix(affinity_matrix)

    embedding_precomp = SpectralEmbedding(
        n_components=2, affinity="precomputed", random_state=42
    )
    out = embedding_precomp.fit_transform(affinity_matrix)
    assert out.shape == (200, 2)
