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
from sklearn.datasets import (
    fetch_openml,
    load_digits,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.manifold import SpectralEmbedding as skSpectralEmbedding
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split

from cuml.manifold import SpectralEmbedding, spectral_embedding

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


def load_mnist(n_samples):
    """Load and sample MNIST dataset."""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(np.int32)
    X = X / 255.0
    X, _, _, _ = train_test_split(
        X, y, train_size=n_samples, stratify=y, random_state=42
    )
    return X


def load_digits_dataset(n_samples=None):
    """Load digits dataset (n_samples is ignored as dataset has fixed size)."""
    digits = load_digits()
    return digits.data


# Dataset configurations: (dataset_loader, dataset_name, n_samples, min_trustworthiness)
dataset_configs = [
    (generate_s_curve, "S-curve", 1500, 0.8),
    (generate_s_curve, "S-curve", 2000, 0.8),
    (generate_swiss_roll, "Swiss Roll", 2000, 0.8),
    (generate_swiss_roll, "Swiss Roll", 3000, 0.8),
    (load_mnist, "MNIST", 5000, 0.8),
    (load_digits_dataset, "Digits dataset", None, 0.8),
]


@pytest.mark.parametrize(
    "dataset_loader,dataset_name,n_samples,min_trustworthiness",
    dataset_configs,
)
def test_spectral_embedding_trustworthiness(
    dataset_loader, dataset_name, n_samples, min_trustworthiness
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

    # Display results
    display_n = n_samples if n_samples else len(X)
    print(f"\n{dataset_name} (n={display_n}):")
    print(f"  sklearn trustworthiness: {trust_sklearn:.4f}")
    print(f"  cuML trustworthiness: {trust_cuml:.4f}")
    print(f"  Difference: {abs(trust_sklearn - trust_cuml):.4f}")

    # Assertions
    assert (
        trust_sklearn > min_trustworthiness
    ), f"sklearn trustworthiness {trust_sklearn:.4f} is too low"
    assert (
        trust_cuml > min_trustworthiness
    ), f"cuML trustworthiness {trust_cuml:.4f} is too low"


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
        ), f"Expected shape ({n_samples}, {n_components}), got {embedding.shape}"

    # Test 2: Reproducibility with seed
    seed = 123
    embedding1 = spectral_embedding(X_gpu, n_components=2, random_state=seed)
    embedding2 = spectral_embedding(X_gpu, n_components=2, random_state=seed)

    assert cp.allclose(
        embedding1, embedding2
    ), "Same seed should produce identical results"

    print(
        "\n✓ spectral_embedding function API tests passed (shape & reproducibility)"
    )
