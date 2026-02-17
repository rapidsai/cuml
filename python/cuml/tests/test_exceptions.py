#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import is_classifier, is_regressor

from cuml.cluster import DBSCAN, HDBSCAN, KMeans
from cuml.decomposition import TruncatedSVD
from cuml.kernel_ridge import KernelRidge  # noqa: F401
from cuml.linear_model import (  # noqa: F401
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from cuml.neighbors import (  # noqa: F401
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)

# Currently only certain estimators raise a NotImplementedError
estimators = {
    "KMeans": lambda: KMeans(n_clusters=2, random_state=0),
    "DBSCAN": lambda: DBSCAN(eps=1.0),
    "TruncatedSVD": lambda: TruncatedSVD(n_components=1, random_state=0),
    "LinearRegression": lambda: LinearRegression(),
    "ElasticNet": lambda: ElasticNet(),
    "Ridge": lambda: Ridge(),
    "Lasso": lambda: Lasso(),
    "HDBSCAN": lambda: HDBSCAN(),
}


@pytest.mark.parametrize("estimator_name", list(estimators.keys()))
def test_sparse_not_implemented_exception(estimator_name):
    X_sparse = csr_matrix([[0, 1], [1, 0]])
    y_class = np.array([0, 1])
    y_reg = np.array([0.0, 1.0])
    estimator = estimators[estimator_name]()
    # Fit or fit_transform depending on the estimator type
    with pytest.raises(NotImplementedError):
        if isinstance(
            estimator, (KMeans, DBSCAN, TruncatedSVD, NearestNeighbors)
        ):
            if hasattr(estimator, "fit_transform"):
                estimator.fit_transform(X_sparse)
            else:
                if isinstance(estimator, KNeighborsClassifier):
                    estimator.fit(X_sparse, y_class)
                else:
                    estimator.fit(X_sparse)
        else:
            # For classifiers and regressors, decide which y to provide
            if is_classifier(estimator):
                estimator.fit(X_sparse, y_class)
            elif is_regressor(estimator):
                estimator.fit(X_sparse, y_reg)
            else:
                # Just in case there's an unexpected type
                estimator.fit(X_sparse)


@pytest.mark.parametrize("array_type", ["numpy", "cupy"])
def test_complex_data_rejected(array_type):
    """Complex dtype inputs should raise ValueError with sklearn-compatible message."""
    # XXX Really we should have a version of the "common tests" from scikit-learn
    # XXX that use cupy arrays instead of numpy arrays.
    if array_type == "numpy":
        X = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    else:
        X = cp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])

    km = KMeans(n_clusters=2)
    with pytest.raises(ValueError, match="Complex data not supported"):
        km.fit(X)
