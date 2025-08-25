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

import cuml.cluster
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("HDBSCAN",)


class HDBSCAN(ProxyBase):
    _gpu_class = cuml.cluster.HDBSCAN
    _not_implemented_attributes = frozenset(
        ("exemplars_", "outlier_scores_", "relative_validity_")
    )
