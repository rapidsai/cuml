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

import warnings

from cuml.internals.import_utils import has_dask

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor", "NearestNeighbors"]

if has_dask():
    from cuml.dask.neighbors.kneighbors_classifier import KNeighborsClassifier
    from cuml.dask.neighbors.kneighbors_regressor import KNeighborsRegressor
    from cuml.dask.neighbors.nearest_neighbors import NearestNeighbors
else:
    warnings.warn(
        "Dask not found. All Dask-based multi-GPU operation is disabled."
    )
    KNeighborsClassifier = None
    KNeighborsRegressor = None
    NearestNeighbors = None
