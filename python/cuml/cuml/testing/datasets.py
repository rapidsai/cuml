# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs as sklearn_make_blobs
from sklearn.datasets import (
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.model_selection import train_test_split

from cuml.internals.array import elements_in_representable_range
from cuml.testing.strategies import (
    combined_datasets_strategy,
    regression_datasets,
    split_datasets,
    standard_classification_datasets,
    standard_datasets,
    standard_regression_datasets,
)


def is_sklearn_compatible_dataset(X_train, X_test, y_train, _=None):
    """Check if a dataset is compatible with scikit-learn's requirements.

    Parameters
    ----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    _ : optional
        Unused parameter for compatibility

    Returns
    -------
    bool
        True if dataset meets scikit-learn's requirements
    """
    return (
        X_train.shape[1] >= 1
        and (X_train > 0).any()
        and (y_train > 0).any()
        and all(
            np.isfinite(x).all()
            for x in (X_train, X_test, y_train)
            if x is not None
        )
    )


def is_cuml_compatible_dataset(X_train, X_test, y_train, _=None):
    """Check if a dataset is compatible with cuML's requirements.

    Parameters
    ----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    _ : optional
        Unused parameter for compatibility

    Returns
    -------
    bool
        True if dataset meets cuML's requirements
    """
    return (
        X_train.shape[0] >= 2
        and X_train.shape[1] >= 1
        and np.isfinite(X_train).all()
        and all(
            elements_in_representable_range(x, np.float32)
            for x in (X_train, X_test, y_train)
            if x is not None
        )
    )


def make_regression_dataset(datatype, nrows, ncols, n_info, **kwargs):
    """Create a regression dataset with specified parameters.

    Parameters
    ----------
    datatype : numpy.dtype
        Data type to cast the arrays to
    nrows : int
        Number of samples
    ncols : int
        Number of features
    n_info : int
        Number of informative features
    **kwargs : dict
        Additional arguments passed to make_regression

    Returns
    -------
    tuple
        Train-test split arrays cast to specified datatype
    """
    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        random_state=0,
        **kwargs,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=10
    )
    return tuple(
        arr.astype(datatype) for arr in (X_train, X_test, y_train, y_test)
    )


def make_classification_dataset(datatype, nrows, ncols, n_info, num_classes):
    """Create a classification dataset with specified parameters.

    Parameters
    ----------
    datatype : numpy.dtype
        Data type to cast the arrays to
    nrows : int
        Number of samples
    ncols : int
        Number of features
    n_info : int
        Number of informative features
    num_classes : int
        Number of classes

    Returns
    -------
    tuple
        Train-test split arrays cast to specified datatype
    """
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_classes=num_classes,
        random_state=0,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=10
    )

    return X_train, X_test, y_train, y_test


def small_regression_dataset(datatype):
    """Create a small regression dataset for testing.

    Parameters
    ----------
    datatype : numpy.dtype
        Data type to cast the arrays to

    Returns
    -------
    tuple
        Train-test split arrays cast to specified datatype
    """
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=10, random_state=10
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    return X_train, X_test, y_train, y_test


def small_classification_dataset(datatype):
    """Create a small classification dataset for testing.

    Parameters
    ----------
    datatype : numpy.dtype
        Data type to cast the arrays to

    Returns
    -------
    tuple
        Train-test split arrays cast to specified datatype
    """
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=10,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    return X_train, X_test, y_train, y_test


def generate_mnist_like_dataset(n_samples, datatype=np.float32):
    """Generate a classification dataset with MNIST-like characteristics.

    This function creates a synthetic dataset similar to MNIST for testing
    purposes, with customizable parameters. The data is normalized to [0, 1]
    range like real MNIST data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    datatype : numpy.dtype, default=np.float32
        Data type to cast the arrays to

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) normalized to [0, 1] range
    """
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=datatype,
        nrows=n_samples,
        ncols=784,  # Same as MNIST features (28x28 pixels)
        n_info=100,  # Number of informative features
        num_classes=10,  # Same as MNIST classes (digits 0-9)
    )

    # Normalize to [0, 1] range like MNIST
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    return X_train, X_test, y_train, y_test


def make_pattern(name, n_samples):
    """Get a specific pattern dataset for clustering and manifold learning.

    Parameters
    ----------
    name : str
        Name of the pattern to generate. One of:
        - 'noisy_circles'
        - 'noisy_moons'
        - 'varied'
        - 'blobs'
        - 'aniso'
        - 'no_structure'
        - 's_curve'
        - 'swiss_roll'
    n_samples : int
        Number of samples to generate

    Returns
    -------
    list
        [data, params] where data is (X, y) and params are clustering parameters
    """
    np.random.seed(0)
    random_state = 170

    if name == "noisy_circles":
        data = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
        params = {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
        }

    elif name == "noisy_moons":
        data = make_moons(n_samples=n_samples, noise=0.05)
        params = {"damping": 0.75, "preference": -220, "n_clusters": 2}

    elif name == "varied":
        data = sklearn_make_blobs(
            n_samples=n_samples,
            cluster_std=[1.0, 2.5, 0.5],
            random_state=random_state,
        )
        params = {"eps": 0.18, "n_neighbors": 2}

    elif name == "blobs":
        data = sklearn_make_blobs(n_samples=n_samples, random_state=8)
        params = {}

    elif name == "aniso":
        X, y = sklearn_make_blobs(
            n_samples=n_samples, random_state=random_state
        )
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)
        params = {"eps": 0.15, "n_neighbors": 2}

    elif name == "no_structure":
        data = np.random.rand(n_samples, 2), None
        params = {}

    elif name == "s_curve":
        from sklearn.datasets import make_s_curve

        data = make_s_curve(n_samples=n_samples, noise=0.05, random_state=42)
        params = {}

    elif name == "swiss_roll":
        from sklearn.datasets import make_swiss_roll

        data = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
        params = {}

    return [data, params]


def with_dtype(data, dtype):
    """Convert dataset arrays to specified dtype.

    Parameters
    ----------
    data : sequence
        Sequence of arrays to convert (e.g. (X, y) or (X_train, X_test, y_train, y_test))
    dtype : numpy.dtype
        Data type to convert arrays to

    Returns
    -------
    tuple
        Sequence with arrays converted to specified dtype
    """
    return tuple(arr.astype(dtype) for arr in data)


def make_text_classification_dataset(
    n_docs=11314,
    n_features=10000,
    n_classes=20,
    avg_nonzero_per_doc=150,
    apply_tfidf=True,
    dtype=np.float64,
    random_state=0,
):
    """Generate a sparse text-like classification dataset.

    This function generates a sparse bag-of-words matrix and target vector
    that mimic the characteristics of text classification datasets (like
    20 newsgroups) after vectorization, using topic-like word distributions.

    Parameters
    ----------
    n_docs : int, default=11314
        Number of documents to generate
    n_features : int, default=10000
        Vocabulary size (number of features)
    n_classes : int, default=20
        Number of classes/topics
    avg_nonzero_per_doc : int, default=150
        Average number of non-zero features per document
    apply_tfidf : bool, default=True
        Whether to apply TF-IDF-like weighting
    dtype : numpy.dtype, default=np.float64
        Data type for the sparse matrix
    random_state : int, default=0
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X, y) where X is a sparse CSR matrix and y is the target array
    """
    rng = np.random.RandomState(random_state)

    # Class labels (balanced)
    y = rng.randint(0, n_classes, size=n_docs)

    # Class-specific word distributions (topic-like)
    class_word_probs = []
    for _ in range(n_classes):
        # Dirichlet distribution to simulate "topic" structure with sparsity
        alpha = np.ones(n_features) * 0.01
        topic = rng.dirichlet(alpha)
        class_word_probs.append(topic)
    class_word_probs = np.vstack(class_word_probs)

    # Generate sparse bag-of-words for each document
    data = []
    rows = []
    cols = []

    for i in range(n_docs):
        label = y[i]
        doc_len = max(1, rng.poisson(avg_nonzero_per_doc))
        word_indices = rng.choice(
            n_features,
            size=doc_len,
            replace=True,
            p=class_word_probs[label],
        )
        unique, counts = np.unique(word_indices, return_counts=True)
        rows.extend([i] * len(unique))
        cols.extend(unique.tolist())
        data.extend(counts.tolist())

    X = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_docs, n_features),
        dtype=dtype,
    )

    if apply_tfidf:
        # Apply TF-IDF-like weighting
        df = (X > 0).sum(axis=0).A1 + 1.0
        idf = np.log((1.0 + n_docs) / df)
        X = X.multiply(idf).tocsr()

    return X, y


__all__ = [
    # Dataset compatibility
    "is_sklearn_compatible_dataset",
    "is_cuml_compatible_dataset",
    # Dataset generation
    "generate_mnist_like_dataset",
    "make_classification",
    "make_classification_dataset",
    "make_pattern",
    "make_regression",
    "make_regression_dataset",
    "make_text_classification_dataset",
    "small_classification_dataset",
    "small_regression_dataset",
    # Dataset strategies
    "combined_datasets_strategy",
    "regression_datasets",
    "split_datasets",
    "standard_classification_datasets",
    "standard_datasets",
    "standard_regression_datasets",
    # Utilities
    "with_dtype",
]
