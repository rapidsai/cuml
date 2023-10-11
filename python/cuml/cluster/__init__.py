#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cuml.internals.device_support import GPU_ENABLED

from cuml.cluster.dbscan import DBSCAN
from cuml.cluster.kmeans import KMeans
from cuml.cluster.hdbscan import HDBSCAN

# TODO: These need to be deprecated and moved to hdbscan namespace
from cuml.cluster.hdbscan.prediction import all_points_membership_vectors
from cuml.cluster.hdbscan.prediction import approximate_predict

if GPU_ENABLED:
    from cuml.cluster.agglomerative import AgglomerativeClustering
