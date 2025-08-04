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
#

import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import load_digits, make_s_curve, make_swiss_roll
from sklearn.manifold import SpectralEmbedding as skSpectralEmbedding
from sklearn.manifold import trustworthiness

from cuml.manifold import SpectralEmbedding, spectral_embedding
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


# Dataset configurations: (dataset_loader, dataset_name, n_samples, min_trustworthiness)
dataset_configs = [
    (generate_s_curve, 1500, 0.8),
    (generate_s_curve, 2000, 0.8),
    (generate_swiss_roll, 2000, 0.8),
    (generate_swiss_roll, 3000, 0.8),
    (load_mnist, 5000, 0.8),
    (load_digits_dataset, None, 0.8),
]


@pytest.mark.parametrize(
    "dataset_loader,n_samples,min_trustworthiness",
    dataset_configs,
)
def test_spectral_embedding_trustworthiness(
    dataset_loader, n_samples, min_trustworthiness
):
    """Test trustworthiness comparison between sklearn and cuML on various datasets."""
    # Load/generate dataset
    X = dataset_loader(n_samples) if n_samples else dataset_loader(None)

    # sklearn embedding
    sk_spectral = skSpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        affinity="nearest_neighbors",
        random_state=42,
        n_jobs=-1,
    )
    X_sklearn = sk_spectral.fit_transform(X)

    # cuML embedding
    X_gpu = cp.asarray(X)
    cuml_spectral = SpectralEmbedding(
        n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS, random_state=42
    )
    X_cuml_gpu = cuml_spectral.fit_transform(X_gpu)
    X_cuml = cp.asnumpy(X_cuml_gpu)

    # Calculate trustworthiness scores
    trust_sklearn = trustworthiness(X, X_sklearn, n_neighbors=N_NEIGHBORS)
    trust_cuml = trustworthiness(X, X_cuml, n_neighbors=N_NEIGHBORS)

    # Assertions
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
