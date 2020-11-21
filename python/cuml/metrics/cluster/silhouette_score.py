import cupy as cp

from cuml.metrics.cluster.silhouette_score_impl import _silhouette_coeff


def silhouette_score(
        X, labels, metric='euclidean', handle=None):
    """Calculate the mean silhouette coefficient for the provided data

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette schore. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """

    return _silhouette_coeff(
        X, labels, metric=metric, handle=handle
    )


def silhouette_samples(X, labels, metric='euclidean', handle=None):
    """Calculate the silhouette coefficient for each sample in the provided data

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette schore. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """

    sil_scores = cp.empty((X.shape[0],), dtype='float64')

    _silhouette_coeff(
        X, labels, metric=metric, sil_scores=sil_scores, handle=handle
    )

    return sil_scores
