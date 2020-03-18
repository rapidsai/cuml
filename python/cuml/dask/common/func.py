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

from cuml.dask.common.utils import get_client
from cuml.dask.common.part_utils import workers_to_parts
from cuml.dask.common.part_utils import hosts_to_parts


def reduce(futures, func, client=None):

    client = get_client(client)

    who_has = client.who_has(futures)

    workers = [who_has[f.key] for f in futures]
    print(str(workers))

    # Merge within each worker
    futures = [client.submit(func, p)
              for w, p in workers_to_parts(workers).items()]

    who_has = client.who_has(futures)
    workers = [who_has[f.key] for f in futures]

    print(str(workers))

    # Merge within each host
    futures = [client.submit(func, p)
              for w, p in hosts_to_parts(workers).items()]

    # Merge across workers
    return tree_reduce(futures, func)


@dask.delayed
def reduce_func_add(a, b): return a + b if b is not None else a


def tree_reduce(delayed_objs, func=reduce_func_add, client=None):
    """
    Performs a binary tree reduce on an associative
    and commutative function in parallel across
    Dask workers.

    TODO: investigate methods for doing intra-node
    before inter-node reductions.
    Ref: https://github.com/rapidsai/cuml/issues/1881

    Parameters
    ----------
    func : Python function or dask.delayed function
        Delayed function to use for reduction. The reduction function
        should be able to handle the case where the second argument is
        None, and should just return a in this case. This is done to
        save memory, rather than having to build an initializer for
        the starting case.
    delayed_objs : array-like of dask.delayed
        Delayed objects to reduce

    Returns
    -------
    reduced_result : dask.delayed
        Delayed object containing the reduced result.
    """

    while len(delayed_objs) > 1:
        new_delayed_objs = []
        n_delayed_objs = len(delayed_objs)
        for i in range(0, n_delayed_objs, 2):
            # add neighbors
            left = delayed_objs[i]
            right = delayed_objs[i+1]\
                if i < n_delayed_objs - 1 else None
            if isinstance(func, dask.delayed):
                obj = func([left, right])
            else:
                obj = client.submit(func, [left, right])
            new_delayed_objs.append(obj)
        delayed_objs = new_delayed_objs

    return first(delayed_objs)
