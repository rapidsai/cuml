import pytest
import numpy as np
from cuml.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score as sk_adjusted_mutual_info_score

def test_adjusted_mutual_info_score_matches_sklearn():
    # Sample labels
    labels_true = np.array([0, 0, 1, 1, 2, 2])
    labels_pred = np.array([1, 1, 0, 0, 2, 2])
    
    # cuML implementation
    res = adjusted_mutual_info_score(labels_true, labels_pred)
    
    # scikit-learn reference
    expected = sk_adjusted_mutual_info_score(labels_true, labels_pred)
    
    # Assert cuML result matches scikit-learn
    assert pytest.approx(res, rel=1e-12) == expected
