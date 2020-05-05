#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cupy as cp


def sorted_unique_labels(*ys):
    """Extract an ordered array of unique labels from one or more dask arrays
    of labels."""
    ys = (cp.unique(y.map_blocks(lambda x: cp.unique(x)).compute())
          for y in ys)
    labels = cp.unique(cp.concatenate(ys))
    return labels
