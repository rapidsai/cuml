import cupy as cp
from cuml.metrics.cluster import mutual_info_score
from cuml.metrics.cluster.contingency_matrix import contingency_matrix

def expected_mutual_information(contingency):
    """
    Compute the expected mutual information (EMI) for a contingency matrix.

    Parameters
    ----------
    contingency : cupy.ndarray of shape (n_classes_true, n_classes_pred)
        Contingency matrix (rows are true labels, columns are predicted labels).

    Returns
    -------
    emi : float
        Expected Mutual Information.
    """
    # Following sklearn.metrics.cluster.expected_mutual_information
    n_samples = contingency.sum()
    pi = contingency.sum(axis=1)
    pj = contingency.sum(axis=0)
    outer = cp.outer(pi, pj)

    # Avoid division by zero
    nz_mask = outer > 0
    log_outer = cp.zeros_like(outer, dtype=cp.float64)
    log_outer[nz_mask] = cp.log(outer[nz_mask])

    emi = (outer / (n_samples ** 2)) * log_outer
    return cp.sum(emi)


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic"):
    """
    GPU version of adjusted_mutual_info_score using CuPy.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
    labels_pred : array-like of shape (n_samples,)
    average_method : str, default='arithmetic'
        Averaging method same as sklearn.

    Returns
    -------
    ami : float
        Adjusted Mutual Information score.
    """
    labels_true = cp.asarray(labels_true)
    labels_pred = cp.asarray(labels_pred)

    contingency = contingency_matrix(labels_true, labels_pred)
    mi = mutual_info_score(labels_true, labels_pred)
    h_true = -cp.sum((contingency.sum(axis=1) / contingency.sum()) *
                     cp.log(contingency.sum(axis=1) / contingency.sum() + 1e-10))
    h_pred = -cp.sum((contingency.sum(axis=0) / contingency.sum()) *
                     cp.log(contingency.sum(axis=0) / contingency.sum() + 1e-10))

    emi = expected_mutual_information(contingency)

    avg = (h_true + h_pred) / 2.0 if average_method == "arithmetic" else cp.sqrt(h_true * h_pred)
    denom = avg - emi

    return float((mi - emi) / denom) if denom != 0 else 1.0
