# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import cuml.internals.logger as logger
import dask

from cuml.dask.common.utils import get_client
from cuml.dask.common.part_utils import hosts_to_parts
from cuml.dask.common.part_utils import workers_to_parts
from dask.delayed import Delayed
from dask.distributed import wait
from toolz import first


def reduce(futures, func, client=None):
    """
    Performs a cluster-wide reduction by first
    running function on worker->host->cluster. This
    function takes locality into account by first
    reducing partitions local to each worker before
    reducing partitions on each host and, finally,
    reducing the partitions across the cluster into
    a single reduced partition.

    Parameters
    ----------

    futures : array-like of dask.Future futures to reduce
    func : Python reduction function accepting list
           of objects to reduce and returning a single
           reduced object.

    client : dask.distributed.Client to use for scheduling

    Returns
    -------

    output : dask.Future a future containing the final reduce
        object.
    """

    client = get_client(client)

    # Make sure input futures have been assigned to worker(s)
    wait(futures)

    for local_reduction_func in [workers_to_parts, hosts_to_parts]:

        who_has = client.who_has(futures)

        workers = [(first(who_has[m.key]), m) for m in futures]
        worker_parts = local_reduction_func(workers)

        # Short circuit when all parts already have preferred
        # locality
        if len(worker_parts) > 1:
            # Local tree reduction for scalability
            futures = client.compute(
                [tree_reduce(p, func) for w, p in worker_parts.items()]
            )

            wait(futures)

    # Merge across workers
    ret = client.compute(tree_reduce(futures, func))
    wait(ret)

    return ret


def tree_reduce(objs, func=sum):
    """
    Performs a binary tree reduce on an associative
    and commutative function in parallel across
    Dask workers. Since this supports dask.delayed
    objects, which have yet been scheduled on workers,
    it does not take locality into account. As a result,
    any local reductions should be performed before
    this function is called.

    Parameters
    ----------
    func : Python function or dask.delayed function
        Function to use for reduction. The reduction function
        acceps a list of objects to reduce as an argument and
        produces a single reduced object
    objs : array-like of dask.delayed or future
           objects to reduce.

    Returns
    -------
    reduced_result : dask.delayed or future
        if func is delayed, the result will be delayed
        if func is a future, the result will be a future
    """

    func = dask.delayed(func) if not isinstance(func, Delayed) else func

    while len(objs) > 1:
        new_objs = []
        n_objs = len(objs)
        for i in range(0, n_objs, 2):
            inputs = dask.delayed(objs[i : i + 2], pure=False)
            obj = func(inputs)
            new_objs.append(obj)
        wait(new_objs)
        objs = new_objs

    logger.debug(str(objs))

    return first(objs)
