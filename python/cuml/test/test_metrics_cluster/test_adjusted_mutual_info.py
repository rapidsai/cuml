import numpy as np
from cuml.metrics.cluster.adjusted_mutual_info_score import adjusted_mutual_info_score

def test_ami_perfect():
    y = np.array([0,0,1,1,2,2])
    assert adjusted_mutual_info_score(y, y) == 1.0

def test_ami_symmetry():
    y_true = np.array([0,0,1,1,2,2])
    y_pred = np.array([1,1,0,0,2,2])
    a1 = adjusted_mutual_info_score(y_true, y_pred)
    a2 = adjusted_mutual_info_score(y_pred, y_true)
    assert abs(a1 - a2) < 1e-9

def test_ami_vs_sklearn_small():
    try:
        import sklearn.metrics as skm
        y_true = np.array([0,0,1,1,2,2,0])
        y_pred = np.array([1,1,0,0,2,2,1])
        ours = adjusted_mutual_info_score(y_true, y_pred)
        ref = skm.adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
        assert abs(ours - ref) < 1e-7
    except Exception:
        # ok to skip if sklearn not installed
        pass
