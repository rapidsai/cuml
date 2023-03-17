#
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.internals.import_utils import has_dask

from cuml.neighbors.nearest_neighbors import NearestNeighbors
from cuml.neighbors.nearest_neighbors import kneighbors_graph
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor
from cuml.neighbors.kernel_density import (
    KernelDensity,
    VALID_KERNELS,
    logsumexp_kernel,
)

VALID_METRICS = {
    "brute": set(
        [
            "l2",
            "euclidean",
            "l1",
            "cityblock",
            "manhattan",
            "taxicab",
            # TODO: add "braycurtis" after https://github.com/rapidsai/raft/issues/1285
            "canberra",
            "minkowski",
            "lp",
            "chebyshev",
            "linf",
            "jensenshannon",
            "cosine",
            "correlation",
            "inner_product",
            "sqeuclidean",
            "haversine",
        ]
    ),
    "rbc": set(["euclidean", "haversine", "l2"]),
    "ivfflat": set(
        [
            "l2",
            "euclidean",
            "sqeuclidean",
            "inner_product",
            "cosine",
            "correlation",
        ]
    ),
    "ivfpq": set(
        [
            "l2",
            "euclidean",
            "sqeuclidean",
            "inner_product",
            "cosine",
            "correlation",
        ]
    ),
}

VALID_METRICS_SPARSE = {
    "brute": set(
        [
            "euclidean",
            "l2",
            "inner_product",
            "l1",
            "cityblock",
            "manhattan",
            "taxicab",
            "canberra",
            "linf",
            "chebyshev",
            "jaccard",
            "minkowski",
            "lp",
            "cosine",
            "hellinger",
        ]
    )
}
