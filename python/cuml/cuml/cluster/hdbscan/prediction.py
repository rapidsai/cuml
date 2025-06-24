# Copyright (c) 2025, NVIDIA CORPORATION.
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
__all__ = (  # noqa: F822
    "all_points_membership_vectors",
    "approximate_predict",
    "membership_vector",
)


def __getattr__(name):
    import warnings

    if name in __all__:
        warnings.warn(
            f"The `cuml.cluster.hdbscan.prediction` namespace is deprecated "
            f"and will be removed in 25.10. Please import {name!r} from "
            f"`cuml.cluster.hdbscan` directly instead.",
            FutureWarning,
        )
        import cuml.cluster.hdbscan as mod

        return getattr(mod, name)
    raise AttributeError(
        "module `cuml.cluster.hdbscan.prediction` has no attribute 'foo'"
    )
