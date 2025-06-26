#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
