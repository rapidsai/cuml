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

import dask
from toolz import first


@dask.delayed
def reduce_func_add(a, b): return a + b


def tree_reduce(delayed_objs, delayed_func=reduce_func_add):
    """
    Performs a binary tree reduce on an associative
    and commutative function in parallel across
    Dask workers.

    TODO: investigate methods for doing intra-node
    before inter-node reductions.

    Parameters
    ----------
    delayed_func : dask.delayed function
        Delayed function to use for reduction
    delayed_objs : array-like of dask.delayed
        Delayed objects to reduce

    Returns
    -------
    reduced_result : dask.delayed
        Delayed object containing the reduced result.
    """
    while len(delayed_objs) > 1:
        new_delayed_objs = []
        for i in range(0, len(delayed_objs), 2):
            # add neighbors
            lazy = delayed_func(delayed_objs[i],
                                delayed_objs[i+1])
            new_delayed_objs.append(lazy)
        delayed_objs = new_delayed_objs

    return first(delayed_objs)
