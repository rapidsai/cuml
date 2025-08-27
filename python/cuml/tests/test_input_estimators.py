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

import inspect
from functools import lru_cache

import numpy as np
import pytest

import cuml
from cuml.datasets import make_regression
from cuml.model_selection import train_test_split
from cuml.testing.utils import ClassEnumerator

###############################################################################
#                              Configurations                                 #
###############################################################################


test_dtypes_all = [
    np.float16,
    np.int16,
]

linear_models_config = ClassEnumerator(module=cuml.linear_model)
models = linear_models_config.get_models()

solver_config = ClassEnumerator(
    module=cuml.solvers,
    # QN uses softmax here because some of the tests uses multiclass
    # logistic regression which requires a softmax loss
    custom_constructors={"QN": lambda: cuml.QN(loss="softmax")},
)
models.update(solver_config.get_models())

cluster_config = ClassEnumerator(
    module=cuml.cluster,
    exclude_classes=[cuml.DBSCAN, cuml.AgglomerativeClustering, cuml.HDBSCAN],
)
models.update(cluster_config.get_models())

decomposition_config = ClassEnumerator(module=cuml.decomposition)
models.update(decomposition_config.get_models())

decomposition_config_xfail = ClassEnumerator(module=cuml.random_projection)
models.update(decomposition_config_xfail.get_models())

neighbor_config = ClassEnumerator(
    module=cuml.neighbors, exclude_classes=[cuml.neighbors.KernelDensity]
)
models.update(neighbor_config.get_models())

models.update({"DBSCAN": cuml.DBSCAN})

models.update({"AgglomerativeClustering": cuml.AgglomerativeClustering})

models.update({"HDBSCAN": cuml.HDBSCAN})

models.update({"UMAP": cuml.UMAP})

k_neighbors_config = ClassEnumerator(
    module=cuml.neighbors,
    exclude_classes=[
        cuml.neighbors.NearestNeighbors,
        cuml.neighbors.KernelDensity,
    ],
)
models.update(k_neighbors_config.get_models())


###############################################################################
#                              Helper Functions                               #
###############################################################################


@lru_cache()
def make_dataset(dtype, nrows, ncols, ninfo):
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=ninfo, random_state=0
    )
    X = X.astype(dtype)
    y = y.astype(dtype).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, y_train, X_test


###############################################################################
#                                  Tests                                      #
###############################################################################


@pytest.mark.parametrize("model_name", models.keys())
@pytest.mark.parametrize("dtype", test_dtypes_all)
def test_estimators_all_dtypes(model_name, dtype):
    nrows = 500
    ncols = 20
    ninfo = 10

    X_train, y_train, X_test = make_dataset(dtype, nrows, ncols, ninfo)
    print(model_name)
    if model_name == "KMeans":
        model = models[model_name](n_init="auto")
    elif model_name in ["SparseRandomProjection", "GaussianRandomProjection"]:
        model = models[model_name](n_components=5)
    else:
        model = models[model_name]()
    sign = inspect.signature(model.fit)
    if "y" in sign.parameters:
        model.fit(X=X_train, y=y_train)
    else:
        model.fit(X=X_train)

    if hasattr(model, "predict"):
        res = model.predict(X_test)

    if hasattr(model, "transform"):
        res = model.transform(X_test)  # noqa: F841
