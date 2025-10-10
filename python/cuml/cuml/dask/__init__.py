# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

try:
    import dask
    import dask.distributed as _  # noqa
    import dask_cudf as _  # noqa
    import raft_dask as _  # noqa

    del _
except ModuleNotFoundError as exc:
    from cupy.cuda import get_local_runtime_version

    ver = f"cu{str(get_local_runtime_version())[:2]}"

    raise ModuleNotFoundError(
        f"{exc!s}\n\n"
        "Not all requirements for using `cuml.dask` are installed.\n\n"
        "Please either conda or pip install as follows:\n\n"
        "# Install with Conda:\n"
        "  conda install rapids-dask-dependency dask-cudf raft-dask\n\n"
        "# Or install with pip:\n"
        f"  pip install cuml-{ver}[dask]"
    ) from exc

from cuml.dask import (
    cluster,
    common,
    datasets,
    decomposition,
    ensemble,
    feature_extraction,
    linear_model,
    manifold,
    metrics,
    naive_bayes,
    neighbors,
    preprocessing,
    solvers,
)

# Avoid "p2p" shuffling in dask for now
dask.config.set({"dataframe.shuffle.method": "tasks"})

__all__ = [
    "cluster",
    "common",
    "datasets",
    "decomposition",
    "ensemble",
    "feature_extraction",
    "linear_model",
    "manifold",
    "metrics",
    "naive_bayes",
    "neighbors",
    "preprocessing",
    "solvers",
]
