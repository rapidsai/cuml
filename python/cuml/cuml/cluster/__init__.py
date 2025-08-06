#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

from cuml.cluster.agglomerative import AgglomerativeClustering
from cuml.cluster.dbscan import DBSCAN
from cuml.cluster.hdbscan import HDBSCAN
from cuml.cluster.kmeans import KMeans


def __getattr__(name):
    import warnings

    if name in ("all_points_membership_vectors", "approximate_predict"):
        warnings.warn(
            f"Accessing {name!r} from the `cuml.cluster` namespace is deprecated "
            "and will be removed in 25.10. Please access it from the "
            "`cuml.cluster.hdbscan` namespace instead.",
            FutureWarning,
        )
        import cuml.cluster.hdbscan as mod

        return getattr(mod, name)
    raise AttributeError(f"module `cuml.cluster` has no attribute {name!r}")
