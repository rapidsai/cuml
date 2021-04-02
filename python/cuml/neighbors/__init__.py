#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cuml.common.import_utils import has_dask

from cuml.neighbors.nearest_neighbors import NearestNeighbors
from cuml.neighbors.nearest_neighbors import kneighbors_graph
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor

VALID_METRICS = {
    "brute": set([
        "l2", "euclidean",
        "l1", "cityblock", "manhattan", "taxicab",
        "braycurtis", "canberra",
        "minkowski", "lp",
        "chebyshev", "linf",
        "jensenshannon",
        "cosine", "correlation",
        "inner_product", "sqeuclidean",
        "haversine"
    ]),
    "ivfflat": set(["l2", "euclidean"]),
    "ivfpq": set(["l2", "euclidean"]),
    "ivfsq": set(["l2", "euclidean"])
    }

VALID_METRICS_SPARSE = {
    "brute": set(["euclidean", "l2", "inner_product",
                  "l1", "cityblock", "manhattan", "taxicab",
                  "canberra", "linf", "chebyshev", "jaccard",
                  "minkowski", "lp", "cosine", "jensenshannon",
                  "russelrao", "kl_divergence", "hamming", "hellinger"])
}
