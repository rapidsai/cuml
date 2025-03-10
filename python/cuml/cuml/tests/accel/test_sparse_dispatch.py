#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import pytest
import numpy as np

from cuml.internals.global_settings import GlobalSettings
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    Lasso,
)
from sklearn.neighbors import (
    NearestNeighbors,
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.base import is_classifier, is_regressor
from hdbscan import HDBSCAN
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
}


@pytest.mark.parametrize("estimator_name", list(estimators.keys()))
def test_sparse_support(estimator_name):
    if not GlobalSettings().accelerator_active and estimator_name == "UMAP":
        pytest.skip(reason="UMAP CPU library fails on this small dataset")
    X_sparse = csr_matrix([[0, 1], [1, 0]])
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
