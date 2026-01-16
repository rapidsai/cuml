import numpy as np
import cupy as cp
import pytest
from sklearn.metrics import adjusted_mutual_info_score as sk_adjusted_mutual_info_score

# Import the cuML implementations we are adding
from _adjusted_mutual_info_score import adjusted_mutual_info_score


def test_adjusted_mutual_info_score():
    """Test adjusted_mutual_info_score against scikit-learn reference values."""
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])

    # Transfer to GPU
    labels_a_gpu = cp.asarray(labels_a)
    labels_b_gpu = cp.asarray(labels_b)

    # Default average_method='arithmetic'
    ami_cuml = adjusted_mutual_info_score(labels_a_gpu, labels_b_gpu)
    ami_sk = sk_adjusted_mutual_info_score(labels_a, labels_b)
    np.testing.assert_allclose(ami_cuml, ami_sk, atol=1e-6)

    # Perfect agreement case (different label values but same partitioning)
    ami_cuml = adjusted_mutual_info_score(cp.array([1, 1, 2, 2]), cp.array([2, 2, 3, 3]))
    assert ami_cuml == pytest.approx(1.0)

    # Large array case (scaled up to test numerical stability)
    a_large = np.tile(labels_a, 110)
    b_large = np.tile(labels_b, 110)
    ami_cuml = adjusted_mutual_info_score(cp.asarray(a_large), cp.asarray(b_large))
    ami_sk = sk_adjusted_mutual_info_score(a_large, b_large)
    np.testing.assert_allclose(ami_cuml, ami_sk, atol=1e-5)

    # Test other average_method values
    for avg in ['min', 'geometric', 'arithmetic', 'max']:
        ami_cuml = adjusted_mutual_info_score(labels_a_gpu, labels_b_gpu, average_method=avg)
        ami_sk = sk_adjusted_mutual_info_score(labels_a, labels_b, average_method=avg)
        np.testing.assert_allclose(ami_cuml, ami_sk, atol=1e-6)
