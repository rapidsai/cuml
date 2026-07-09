# Modified test_metrics.py to test label permutation invariance
import numpy as np
from cuml.metrics import homogeneity_score, completeness_score, v_measure_score, mutual_info_score
from cuml.common.utils import sort_labels

def test_label_permutation_invariance():
    """Test label permutation invariance"""
    y_true = np.array([0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0])
    sorted_labels = sort_labels(y_true)
    sorted_pred_labels = sort_labels(y_pred)
    # Test homogeneity score
    s1 = homogeneity_score(y_true, y_pred)
    s2 = homogeneity_score(y_true, 1 - y_pred)
    assert np.isclose(s1, s2)
    # Test completeness score
    s1 = completeness_score(y_true, y_pred)
    s2 = completeness_score(y_true, 1 - y_pred)
    assert np.isclose(s1, s2)
    # Test v-measure score
    s1 = v_measure_score(y_true, y_pred)
    s2 = v_measure_score(y_true, 1 - y_pred)
    assert np.isclose(s1, s2)
    # Test mutual information score
    s1 = mutual_info_score(y_true, y_pred)
    s2 = mutual_info_score(y_true, 1 - y_pred)
    assert np.isclose(s1, s2)

if __name__ == "__main__":
    test_label_permutation_invariance()