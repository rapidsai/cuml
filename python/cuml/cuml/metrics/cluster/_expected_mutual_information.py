import cupy as cp
from cuml.metrics.cluster import mutual_info_score

from scipy.special import gammaln
from math import exp, lgamma

def expected_mutual_information(contingency: cp.ndarray, n_samples: int):
    """
    Calculate the expected mutual information given contingency matrix and number of samples.

    Parameters
    ----------
    contingency: cupy.ndarray
        Contingency matrix of shape (n_class_true, n_class_pred).

    Returns
    -------
    emi: float
        Expected mutual information.
    """
    emi = 0.0
    n_rows, n_cols = contingency.shape
    a = cp.ravel(contingency.sum(axis=1).astype(cp.int64, copy=False))
    b = cp.ravel(contingency.sum(axis=0).astype(cp.int64, copy=False))
    a_view = a
    b_view = b

    # any labelling with zero entropy implies EMI = 0
    if a.size == 1 or b.size == 1:
        return 0.0

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = cp.arange(0, max(cp.max(a), cp.max(b)) + 1, dtype='float')
    nijs[0] = 1 # Stops divide by zero warnings. As its not used, no issue.
    # term1 is nij/N
    term1 = nijs / n_samples
    # term2 = log((N * nij) / (a * b)) = log(N * nij) - log(a * b)
    log_a = cp.log(a)
    log_b = cp.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = cp.log(n_samples) + cp.log(nijs)
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(n_samples - a + 1)
    gln_Nb = gammaln(n_samples - b + 1)
    gln_Nnij = gammaln(nijs + 1) + gammaln(n_samples + 1)

    # emi itslef is a summation over the various values.
    for i in range(n_rows):
        for j in range(n_cols):
            start = int(max(1, a_view[i] - n_samples + b_view[j]))
            end = int(min(a_view[i], b_view[j]) + 1)
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                       - gln_Nnij[nij] - lgamma(a_view[i] - nij + 1)
                       - lgamma(b_view[j] - nij + 1)
                       - lgamma(n_samples - a_view[i] - b_view[j] + nij + 1))
                term3 = exp(gln)
                emi += (term1[nij] * term2 * term3)
    return float(emi)
