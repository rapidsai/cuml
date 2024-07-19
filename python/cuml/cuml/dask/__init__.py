# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from dask import config

from cuml.dask import cluster
from cuml.dask import common
from cuml.dask import datasets
from cuml.dask import decomposition
from cuml.dask import ensemble
from cuml.dask import feature_extraction
from cuml.dask import linear_model
from cuml.dask import manifold
from cuml.dask import metrics
from cuml.dask import naive_bayes
from cuml.dask import neighbors
from cuml.dask import preprocessing
from cuml.dask import solvers

# Avoid "p2p" shuffling in dask for now
config.set({"dataframe.shuffle.method": "tasks"})

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
