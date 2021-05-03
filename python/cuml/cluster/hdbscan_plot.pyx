#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

# distutils: language = c++

from libc.stdint cimport uintptr_t

import numpy as np

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.mixins import ClusterMixin
from cuml.common.mixins import CMajorInputTagMixin

from cuml.metrics.distance_type cimport DistanceType

class MinimumSpanningTree:

    # Accepts CumlArray objects for MST parts
    def __init__(self, mst_src, mst_dst, mst_weights):
        pass

class CondensedTree:

    # TODO: CondensedTree: Expose C++ function `select_clusters()`
    #  which calls excess of mass or leaf method accordingly.

    # Accepts CumlArray object for condensed tree parts
    def __init__(self, parents, children, lambdas, sizes):
        pass

class SingleLinkageTree:


    # Accepts CumlArray objects for dendrogram parts
    def __init__(self, children, deltas, sizes):
        pass
