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

except ImportError as exc:
    msg = (
        f"{exc!s}\n\n"
        "Not all requirements for using `cuml.dask` are installed.\n\n"
        "Please either conda or pip install as follows:\n\n"
        "  conda install cuml-dask      # either conda install\n"
        '  pip install -U "cuml[dask]"  # or pip install\n'
    )
    raise ImportError(msg) from exc

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
