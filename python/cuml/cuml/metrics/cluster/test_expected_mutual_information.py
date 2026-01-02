import cupy as cp
import numpy as np
from _expected_mutual_information import expected_mutual_information
from sklearn.metrics.cluster import expected_mutual_information as sk_expected_mutual_information
from sklearn.metrics.cluster import contingency_matrix

def test_expected_mutual_information():
    """Test expected_mutual_information against scikit-learn reference."""
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])

    # Compute contingency matrix on CPU (dense)
    contingency = contingency_matrix(labels_a, labels_b)  # numpy array
    n_samples = contingency.sum()

    # Transfer contingency to GPU
    contingency_gpu = cp.asarray(contingency)

    emi_cuml = expected_mutual_information(contingency_gpu, n_samples)
    emi_sk = sk_expected_mutual_information(contingency, n_samples)

    np.testing.assert_allclose(emi_cuml, emi_sk, atol=1e-6)

    # Overflow regression test (large single cell)
    large_contingency = np.array([[70000]])
    large_contingency_gpu = cp.asarray(large_contingency)
    emi_large = expected_mutual_information(large_contingency_gpu,n_samples = large_contingency.sum())
    assert emi_large <= 1.0

    # Additional small cases for coverage
    small_cases = [
        ([[1, 0], [0, 1]], 2),        # two distinct clusters
        ([[5, 5], [5, 5]], 20),        # random-like
        ([[10]], 10),                 # single cluster
    ]
    for cont, n in small_cases:
        cont_np = np.asarray(cont)
        cont_gpu = cp.asarray(cont_np)
        emi_cuml = expected_mutual_information(cont_gpu,n_samples=cont_gpu.sum())
        emi_sk = sk_expected_mutual_information(cont_np, n)
        np.testing.assert_allclose(emi_cuml, emi_sk, atol=1e-8)
