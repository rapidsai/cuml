import cupy as cp
from _expected_mutual_information import expected_mutual_information
from scipy import sparse as sp
from cuml.metrics.cluster import entropy
from cupyx.scipy import sparse as cp_sparse

def check_clustering(labels_true, labels_pred):
    """
    """
    labels_true = cp.asarray(labels_true)
    labels_pred = cp.asarray(labels_pred)

    if labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise ValueError("labels_true and labels_pred must be 1-D arrays")
    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have same length")

    # Optional: Handle continuous label values

    return labels_true, labels_pred

# @validate_params(
#     {
#         "labels_true": ["array-like", None],
#         "labels_pred": ["array-like", None],
#         "eps": [Interval(Real, 0, None, closed="left"), None],
#         "sparse": ["boolean"],
#         "dtype": "no_validation",  # delegate the validation to SciPy
#     },
#     prefer_skip_nested_validation=True,
#     )
def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype=cp.float64):
    """
    """
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = cp.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = cp.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    # Using coo_matrix to calculate contingency matrix from scipy.sparse
    contingency = cp_sparse.coo_matrix(
        (cp.ones(class_idx.shape[0], dtype=cp.float64), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=cp.float64,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
        return contingency.toarray() # Returns contingecy as a cupy.ndarray
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contigency is integer
            contingency = contingency + eps

        # return contingency
    return contingency

def _generalized_average(U,V,average_method):
    """ Return a particular mean of two numbers"""
    if average_method == "min":
        return min(U,V)
    elif average_method == "max":
        return max(U,V)
    elif average_method == "geometric":
        return cp.sqrt(U * V)
    elif average_method == "arithmetic":
        return cp.mean(cp.asarray([U,V]))
    else:
        raise ValueError(
            "'average_method' must be 'arithmetic', 'max', 'min' or 'geometric'"
        )

def _entropy(labels):
    # cuml.metrics.entropy has a strict type requirement as int32
    if labels.dtype != cp.int32:
        labels = labels.astype(cp.int32)
    return entropy(labels)

def raw_mutual_info_score(contingency):
    """ Compute raw (unnormalized) mutual information from contingency matrix.
    Exact CuPy translation of sklearn's implementation."""
    contingency = cp.asarray(contingency, dtype=cp.float64)
    contingency_sum = contingency.sum()
    if contingency_sum == 0 or contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
        return 0.0

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    nz_idx = contingency > 0
    nij_nz = contingency[nz_idx]

    expected = (row_sums[:, None] @ col_sums[None, :])[nz_idx]

    mi = cp.sum(
        (nij_nz / contingency_sum) * cp.log((nij_nz * contingency_sum) / expected)
    )
    return float(cp.maximum(mi, 0.0))


# @validate_params(
#     {
#         "labels_true": ["array-like"],
#         "labels_pred": ["array-like"],
#         "average_method": [StrOptions({"arthematic", "max", "min", "geometric"})]

#     },
#     prefer_skip_nested_validation=True,
# )

def adjusted_mutual_info_score(
        labels_true, labels_pred, *, average_method="arithmetic"
):
    """Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (``label_true``)
    with :math:`V` (``labels_pred``) will return the same score value. This can
    be useful to measure the agreement of two independent label assignments
    strategies on the same dataset when the real ground truth is not known.

    Be mindful that this function is an order of magnitude slower than other
    metrics, such as the Adjusted Rand Index.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.
    Returns
    -------
    ami: float (upperlimited by 1.0)
       The AMI returns a value of 1 when the two partitions are identical
       (ie perfectly matched). Random partitions (independent labellings) have
       an expected AMI around 0 on average hence can be negative. The value is
       in adjusted nats (based on the natural logarithm).
    """
    labels_true, labels_pred = check_clustering(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = cp.unique(labels_true)
    clusters = cp.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0
    # if there is only one class or one cluster return 0.0.
    elif classes.shape[0] == 1 or clusters.shape[0] == 1:
        return 0.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calcualte the MI for the two clusterings
    mi = raw_mutual_info_score(contingency)
    # Calculate the expected value for the mutual information
    emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = _entropy(labels_true), _entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e. a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -cp.finfo("float64").eps)
    else:
        denominator = max(denominator, cp.finfo("float64").eps)
    # The same applies analogously to mi and emi.
    numerator = mi - emi
    if numerator < 0:
        numerator = min(numerator, -cp.finfo("float64").eps)
    else:
        numerator = max(numerator, cp.finfo("float64").eps)
    return float(numerator / denominator)
