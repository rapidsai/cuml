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

import os
import h5py
import cupy as cp
import numpy as np
import pytest
from urllib.request import urlretrieve
from sklearn import datasets
from sklearn.datasets import fetch_openml, make_s_curve
from sklearn.manifold import SpectralEmbedding as skSpectralEmbedding
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split

from cuml.manifold import SpectralEmbedding
from cuml.metrics import trustworthiness as cuml_trustworthiness

# Test parameters
N_NEIGHBORS = 15
N_COMPONENTS = 2
TRUSTWORTHINESS_TOLERANCE = 0.05  # Maximum allowed difference between sklearn and cuML


def download_fashion_mnist(cache_dir='/tmp/rapids_datasets'):
    """Download Fashion-MNIST dataset if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, "fashion-mnist-784-euclidean.hdf5")
    url = 'https://data.rapids.ai/cuvs/datasets/fashion-mnist-784-euclidean.hdf5'
    
    if not os.path.exists(filepath):
        print("Downloading Fashion-MNIST dataset...")
        urlretrieve(url, filepath)
    
    return filepath


def load_fashion_mnist_data(max_samples=5000):
    """Load Fashion-MNIST data from HDF5 file."""
    filepath = download_fashion_mnist()
    
    with h5py.File(filepath, 'r') as f:
        # Load the data
        if 'train' in f:
            data = np.array(f['train'])
        elif 'dataset' in f:
            data = np.array(f['dataset'])
        else:
            keys = list(f.keys())
            for key in keys:
                if isinstance(f[key], h5py.Dataset):
                    data = np.array(f[key])
                    break
        
        # Sample if needed
        if max_samples and len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]
    
    return data


@pytest.mark.parametrize("n_samples", [1500, 2000])
def test_spectral_embedding_trustworthiness_s_curve(n_samples):
    """Test trustworthiness comparison between sklearn and cuML on S-curve dataset."""
    # Generate S-curve dataset
    X, color = make_s_curve(n_samples=n_samples, noise=0.05, random_state=42)
    
    # sklearn embedding
    sk_spectral = skSpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        affinity='nearest_neighbors',
        random_state=42,
        n_jobs=-1
    )
    X_sklearn = sk_spectral.fit_transform(X)
    
    # cuML embedding
    X_gpu = cp.asarray(X)
    cuml_spectral = SpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        random_state=42
    )
    X_cuml_gpu = cuml_spectral.fit_transform(X_gpu)
    X_cuml = cp.asnumpy(X_cuml_gpu)
    
    # Calculate trustworthiness scores
    trust_sklearn = trustworthiness(X, X_sklearn, n_neighbors=N_NEIGHBORS)
    trust_cuml = trustworthiness(X, X_cuml, n_neighbors=N_NEIGHBORS)
    
    print(f"\nS-curve (n={n_samples}):")
    print(f"  sklearn trustworthiness: {trust_sklearn:.4f}")
    print(f"  cuML trustworthiness: {trust_cuml:.4f}")
    print(f"  Difference: {abs(trust_sklearn - trust_cuml):.4f}")
    
    # Assert that trustworthiness scores are similar
    assert abs(trust_sklearn - trust_cuml) < TRUSTWORTHINESS_TOLERANCE, \
        f"Trustworthiness difference {abs(trust_sklearn - trust_cuml):.4f} exceeds tolerance {TRUSTWORTHINESS_TOLERANCE}"
    
    # Both should have good trustworthiness (> 0.8 for S-curve)
    assert trust_sklearn > 0.8, f"sklearn trustworthiness {trust_sklearn:.4f} is too low"
    assert trust_cuml > 0.8, f"cuML trustworthiness {trust_cuml:.4f} is too low"


@pytest.mark.parametrize("n_samples", [5000])
def test_spectral_embedding_trustworthiness_mnist(n_samples):
    """Test trustworthiness comparison between sklearn and cuML on MNIST dataset."""
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(np.int32)
    
    # Normalize and sample
    X = X / 255.0
    X, _, y, _ = train_test_split(X, y, train_size=n_samples, 
                                  stratify=y, random_state=42)
    
    # sklearn embedding
    sk_spectral = skSpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        affinity='nearest_neighbors',
        random_state=42,
        n_jobs=-1
    )
    X_sklearn = sk_spectral.fit_transform(X)
    
    # cuML embedding
    X_gpu = cp.asarray(X)
    cuml_spectral = SpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        random_state=42
    )
    X_cuml_gpu = cuml_spectral.fit_transform(X_gpu)
    X_cuml = cp.asnumpy(X_cuml_gpu)
    
    # Calculate trustworthiness scores
    trust_sklearn = trustworthiness(X, X_sklearn, n_neighbors=N_NEIGHBORS)
    trust_cuml = trustworthiness(X, X_cuml, n_neighbors=N_NEIGHBORS)
    
    print(f"\nMNIST (n={n_samples}):")
    print(f"  sklearn trustworthiness: {trust_sklearn:.4f}")
    print(f"  cuML trustworthiness: {trust_cuml:.4f}")
    print(f"  Difference: {abs(trust_sklearn - trust_cuml):.4f}")
    
    # Assert that trustworthiness scores are similar
    assert abs(trust_sklearn - trust_cuml) < TRUSTWORTHINESS_TOLERANCE, \
        f"Trustworthiness difference {abs(trust_sklearn - trust_cuml):.4f} exceeds tolerance {TRUSTWORTHINESS_TOLERANCE}"
    
    # Both should have reasonable trustworthiness (> 0.7 for MNIST)
    assert trust_sklearn > 0.7, f"sklearn trustworthiness {trust_sklearn:.4f} is too low"
    assert trust_cuml > 0.7, f"cuML trustworthiness {trust_cuml:.4f} is too low"


@pytest.mark.parametrize("n_samples", [5000])
def test_spectral_embedding_trustworthiness_fashion_mnist(n_samples):
    """Test trustworthiness comparison between sklearn and cuML on Fashion-MNIST dataset."""
    # Load Fashion-MNIST data
    X = load_fashion_mnist_data(max_samples=n_samples)
    
    # sklearn embedding
    sk_spectral = skSpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        affinity='nearest_neighbors',
        random_state=42,
        n_jobs=-1
    )
    X_sklearn = sk_spectral.fit_transform(X)
    
    # cuML embedding
    X_gpu = cp.asarray(X)
    cuml_spectral = SpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        random_state=42
    )
    X_cuml_gpu = cuml_spectral.fit_transform(X_gpu)
    X_cuml = cp.asnumpy(X_cuml_gpu)
    
    # Calculate trustworthiness scores
    trust_sklearn = trustworthiness(X, X_sklearn, n_neighbors=N_NEIGHBORS)
    trust_cuml = trustworthiness(X, X_cuml, n_neighbors=N_NEIGHBORS)
    
    print(f"\nFashion-MNIST (n={n_samples}):")
    print(f"  sklearn trustworthiness: {trust_sklearn:.4f}")
    print(f"  cuML trustworthiness: {trust_cuml:.4f}")
    print(f"  Difference: {abs(trust_sklearn - trust_cuml):.4f}")
    
    # Assert that trustworthiness scores are similar
    assert abs(trust_sklearn - trust_cuml) < TRUSTWORTHINESS_TOLERANCE, \
        f"Trustworthiness difference {abs(trust_sklearn - trust_cuml):.4f} exceeds tolerance {TRUSTWORTHINESS_TOLERANCE}"
    
    # Both should have reasonable trustworthiness (> 0.7 for Fashion-MNIST)
    assert trust_sklearn > 0.7, f"sklearn trustworthiness {trust_sklearn:.4f} is too low"
    assert trust_cuml > 0.7, f"cuML trustworthiness {trust_cuml:.4f} is too low"


@pytest.mark.benchmark
def test_spectral_embedding_performance_comparison():
    """Benchmark performance comparison between sklearn and cuML."""
    import time
    
    # Generate a medium-sized dataset
    X, _ = make_s_curve(n_samples=5000, noise=0.05, random_state=42)
    
    # Time sklearn
    sk_spectral = skSpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        affinity='nearest_neighbors',
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    _ = sk_spectral.fit_transform(X)
    sklearn_time = time.time() - start_time
    
    # Time cuML (with warm-up)
    X_gpu = cp.asarray(X)
    
    # Warm-up
    cuml_warmup = SpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        random_state=42
    )
    _ = cuml_warmup.fit_transform(X_gpu)
    
    # Actual timing
    cuml_spectral = SpectralEmbedding(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        random_state=42
    )
    
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    _ = cuml_spectral.fit_transform(X_gpu)
    cp.cuda.Stream.null.synchronize()
    cuml_time = time.time() - start_time
    
    speedup = sklearn_time / cuml_time
    
    print(f"\nPerformance Comparison (5000 samples):")
    print(f"  sklearn time: {sklearn_time:.3f}s")
    print(f"  cuML time: {cuml_time:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")
    
    # cuML should be faster
    assert cuml_time < sklearn_time, "cuML should be faster than sklearn"
    
    # For this size, we expect at least 2x speedup
    assert speedup > 2.0, f"Expected at least 2x speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 