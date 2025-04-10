# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import lru_cache

from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.array import elements_in_representable_range
from cuml.testing.strategies import (
    standard_datasets,
    standard_regression_datasets,
    standard_classification_datasets,
    split_datasets,
    regression_datasets,
    combined_datasets_strategy,
)
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_circles,
    make_moons,
    make_blobs as sklearn_make_blobs,
)
from sklearn.model_selection import train_test_split

np = cpu_only_import("numpy")
skl_make_reg = make_regression  # Alias for create_synthetic_dataset


def sklearn_compatible_dataset(X_train, X_test, y_train, _=None):
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


def cuml_compatible_dataset(X_train, X_test, y_train, _=None):
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


def _generate_regression_data(nrows, ncols, n_info, **kwargs):
    """Generate regression data using sklearn's make_regression.

    Parameters
    ----------
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
        Train-test split of generated data
    """
    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        random_state=0,
        **kwargs,
    )
    return train_test_split(X, y, train_size=0.8, random_state=10)


@lru_cache(maxsize=4)  # Cache up to 4 dataset generations
def _get_cached_regression_data(nrows, ncols, n_info, **kwargs):
    """Cached version of _generate_regression_data."""
    return _generate_regression_data(nrows, ncols, n_info, **kwargs)


def make_regression_dataset(datatype, nrows, ncols, n_info, **kwargs):
    """Create a regression dataset with specified parameters.

    Uses caching for datasets under 4GB total size.

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
    # Use cached version for datasets under 4GB
    if nrows * ncols < 1e8:
        dataset = _get_cached_regression_data(nrows, ncols, n_info, **kwargs)
    else:
        dataset = _generate_regression_data(nrows, ncols, n_info, **kwargs)

    return map(lambda arr: arr.astype(datatype), dataset)


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


def create_synthetic_dataset(
    generator=skl_make_reg,
    n_samples=100,
    n_features=10,
    test_size=0.25,
    random_state_generator=None,
    random_state_train_test_split=None,
    dtype=np.float32,
    **kwargs,
):
    """Create a synthetic dataset using the specified generator.

    Parameters
    ----------
    generator : callable, default=skl_make_reg
        Function to generate the dataset
    n_samples : int, default=100
        Number of samples
    n_features : int, default=10
        Number of features
    test_size : float, default=0.25
        Fraction of dataset to use for testing
    random_state_generator : int, optional
        Random state for dataset generation
    random_state_train_test_split : int, optional
        Random state for train-test split
    dtype : numpy.dtype, default=np.float32
        Data type to cast the arrays to
    **kwargs : dict
        Additional arguments passed to the generator

    Returns
    -------
    tuple
        Train-test split arrays cast to specified datatype
    """
    X, y = generator(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state_generator,
        **kwargs,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_train_test_split
    )

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return X_train, X_test, y_train, y_test


def make_pattern_dataset(name, n_samples):
    """Get a specific pattern dataset for clustering.

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

    return [data, params]


__all__ = [
    # Dataset strategies
    "standard_datasets",
    "standard_regression_datasets",
    "standard_classification_datasets",
    "split_datasets",
    "regression_datasets",
    "combined_datasets_strategy",
    # Dataset generation
    "make_regression_dataset",
    "make_classification_dataset",
    "make_regression",
    "make_classification",
    "create_synthetic_dataset",
    "small_regression_dataset",
    "small_classification_dataset",
    # Dataset compatibility
    "sklearn_compatible_dataset",
    "cuml_compatible_dataset",
    # Utilities
    "make_pattern_dataset",
]
