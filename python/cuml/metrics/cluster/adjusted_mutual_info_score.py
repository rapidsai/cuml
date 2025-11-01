# python/cuml/metrics/cluster/adjusted_mutual_info_score.py
"""
Adjusted Mutual Information (AMI) for cuML (numpy/cupy-compatible).
Basic pure-Python implementation modeled after scikit-learn behavior.
"""

from __future__ import annotations
import math
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

def _get_array_module(x):
    if cp is not None:
        try:
            import cupy
            if isinstance(x, cupy.ndarray):
                return cp
        except Exception:
            pass
    return np

def _contingency_matrix(labels_true, labels_pred, xp=np):
    labels_true = xp.asarray(labels_true).ravel()
    labels_pred = xp.asarray(labels_pred).ravel()
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have same size")

    uniq_true, inv_true = xp.unique(labels_true, return_inverse=True)
    uniq_pred, inv_pred = xp.unique(labels_pred, return_inverse=True)
    n_true = uniq_true.shape[0]
    n_pred = uniq_pred.shape[0]

    contingency = xp.zeros((n_true, n_pred), dtype=xp.int64)
    # accumulate counts
    for t, p in zip(inv_true.tolist(), inv_pred.tolist()):
        contingency[int(t), int(p)] += 1
    return contingency

def _mutual_info(contingency, xp=np):
    contingency = xp.asarray(contingency, dtype=xp.float64)
    n_samples = contingency.sum()
    if n_samples == 0:
        return 0.0
    pi = contingency.sum(axis=1)
    pj = contingency.sum(axis=0)

    nzx, nzy = xp.nonzero(contingency)
    mi = 0.0
    for i, j in zip(nzx.tolist(), nzy.tolist()):
        nij = contingency[int(i), int(j)]
        if nij > 0:
            mi += (nij / n_samples) * math.log((nij * n_samples) / (pi[int(i)] * pj[int(j)]))
    return float(mi)

def _entropy(counts, xp=np):
    counts = xp.asarray(counts, dtype=xp.float64)
    n = counts.sum()
    if n == 0:
        return 0.0
    probs = counts[counts > 0] / n
    return float(-xp.sum(probs * xp.log(probs)))

def _expected_mutual_info(contingency):
    # This uses a combinatorial summation to approximate/compute EMI.
    # Implemented in Python for correctness over raw speed.
    contingency = np.asarray(contingency, dtype=np.int64)
    n = int(contingency.sum())
    if n == 0:
        return 0.0

    a = contingency.sum(axis=1).astype(int).tolist()
    b = contingency.sum(axis=0).astype(int).tolist()
    from math import lgamma

    def comb(n_, k_):
        # C(n, k) computed via gamma for stability
        return math.exp(lgamma(n_ + 1) - lgamma(k_ + 1) - lgamma(n_ - k_ + 1))

    emi = 0.0
    for ai in a:
        for bj in b:
            min_nij = max(1, ai + bj - n)
            max_nij = min(ai, bj)
            if max_nij < min_nij:
                continue
            term_sum = 0.0
            for nij in range(min_nij, max_nij + 1):
                p = comb(ai, nij) * comb(n - ai, bj - nij) / comb(n, bj)
                if p > 0 and nij > 0:
                    term = (nij / float(n)) * math.log((nij * float(n)) / (ai * bj))
                    term_sum += p * term
            emi += term_sum
    return float(emi)

def adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'):
    """
    Compute the Adjusted Mutual Information (AMI) score between two cluster label assignments.

    The AMI measures the agreement between two clustering results while adjusting
    for chance. It ranges from 0 (random labeling) to 1 (perfect agreement).

    Parameters
    ----------
    labels_true : array-like
        Ground truth class labels.
    labels_pred : array-like
        Predicted cluster labels to evaluate.
    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer used in the denominator.

    Returns
    -------
    float
        The Adjusted Mutual Information score.

    Notes
    -----
    This implementation aligns with the scikit-learn version of
    ``sklearn.metrics.adjusted_mutual_info_score``.

    Examples
    --------
    >>> from cuml.metrics.cluster import adjusted_mutual_info_score
    >>> import numpy as np
    >>> labels_true = np.array([0, 0, 1, 1])
    >>> labels_pred = np.array([1, 1, 0, 0])
    >>> adjusted_mutual_info_score(labels_true, labels_pred)
    1.0
    """

    xp = _get_array_module(labels_true)
    contingency = _contingency_matrix(labels_true, labels_pred, xp=xp)
    # compute MI and entropies (use xp for arrays)
    mi = _mutual_info(contingency, xp=xp)
    h_true = _entropy(contingency.sum(axis=1), xp=xp)
    h_pred = _entropy(contingency.sum(axis=0), xp=xp)

    # compute EMI with numpy (safer for combinatorics)
    emi = _expected_mutual_info(np.asarray(contingency))

    if average_method == 'min':
        avg_h = min(h_true, h_pred)
    elif average_method == 'max':
        avg_h = max(h_true, h_pred)
    elif average_method == 'geometric':
        avg_h = math.sqrt(h_true * h_pred)
    else:
        avg_h = 0.5 * (h_true + h_pred)

    denom = avg_h - emi
    if denom <= 0:
        # if MI equals EMI (or denom non-positive) follow sklearn behavior:
        return 1.0 if (mi - emi) == 0 else 0.0
    ami = (mi - emi) / denom
    return float(ami)

__all__ = ["adjusted_mutual_info_score"]
