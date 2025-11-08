#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from hdbscan import HDBSCAN
from scipy.sparse import csr_matrix
from sklearn.base import is_classifier, is_regressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from umap import UMAP

estimators = {
    "KMeans": lambda: KMeans(n_clusters=2, random_state=0),
    "DBSCAN": lambda: DBSCAN(eps=1.0),
    "TruncatedSVD": lambda: TruncatedSVD(n_components=1, random_state=0),
    "LinearRegression": lambda: LinearRegression(),
    "LogisticRegression": lambda: LogisticRegression(),
    "ElasticNet": lambda: ElasticNet(),
    "Ridge": lambda: Ridge(),
    "Lasso": lambda: Lasso(),
    "NearestNeighbors": lambda: NearestNeighbors(n_neighbors=1),
    "UMAP": lambda: UMAP(n_components=1),
    "HDBSCAN": lambda: HDBSCAN(),
    "SVC": lambda: SVC(),
    "SVR": lambda: SVR(),
    "KernelRidge": lambda: KernelRidge(),
    "LinearSVC": lambda: LinearSVC(),
    "LinearSVR": lambda: LinearSVR(),
}


@pytest.mark.parametrize("estimator_name", list(estimators.keys()))
def test_sparse_support(estimator_name):
    X_sparse = csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    y_class = np.array([0, 1])
    y_reg = np.array([0.0, 1.0])
    estimator = estimators[estimator_name]()
    # Fit or fit_transform depending on the estimator type
    if isinstance(estimator, (KMeans, DBSCAN, TruncatedSVD, NearestNeighbors)):
        if hasattr(estimator, "fit_transform"):
            estimator.fit_transform(X_sparse)
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
