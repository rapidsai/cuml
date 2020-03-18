# Copyright (c) 2019, NVIDIA CORPORATION.
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

from uuid import uuid1
from collections import OrderedDict

from functools import reduce

from cuml.dask.common.utils import parse_host_port


def hosts_to_parts(futures):
    """
    Builds an ordered dict mapping each host to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        print(str(w))
        host, port = parse_host_port(w)
        host_key = [host, port]
        if host_key not in w_to_p_map:
            w_to_p_map[host_key] = []
        w_to_p_map[host_key].append(p)
    return w_to_p_map


def workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append(p)
    return w_to_p_map


def _func_get_rows(df):
    return df.shape[0]


def parts_to_ranks(client, worker_info, part_futures):
    """
    Builds a list of (rank, size) tuples of partitions
    :param worker_info: dict of {worker, {"r": rank }}. Note: \
        This usually comes from the underlying communicator
    :param part_futures: list of (worker, future) tuples
    :return: [(part, size)] in the same order of part_futures
    """
    key = uuid1()
    futures = [(worker_info[wf[0]]["r"],
                client.submit(_func_get_rows,
                              wf[1],
                              workers=[wf[0]],
                              key="%s-%s" % (key, idx)))
               for idx, wf in enumerate(part_futures)]

    sizes = client.compute(list(map(lambda x: x[1], futures)), sync=True)
    total = reduce(lambda a, b: a + b, sizes)

    return [(futures[idx][0], size) for idx, size in enumerate(sizes)], total


def _default_part_getter(f, idx): return f[idx]


def flatten_grouped_results(client, gpu_futures,
                            worker_results_map,
                            getter_func=_default_part_getter):
    """
    This function is useful when a series of partitions have been grouped by
    the worker responsible for the data and the resulting partitions are
    stored on each worker as a list. This happens when a communications
    implementation is used which does not allow multiple ranks per device, so
    the partitions need to be grouped on the ranks to be processed concurrently
    using different streams.

    :param client: Dask client
    :param gpu_futures: [(future, part)] worker to part list of tuples
    :param worker_results_map: { rank: future } where future is a list
           of data partitions on a Dask worker
    :param getter_func: a function that takes a future and partition index
           as arguments and returns the data for a specific partitions
    :return: the ordered list of futures holding each partition on the workers
    """
    futures = []
    completed_part_map = {}
    for rank, part in gpu_futures:
        if rank not in completed_part_map:
            completed_part_map[rank] = 0

        f = worker_results_map[rank]

        futures.append(client.submit(
            getter_func, f, completed_part_map[rank]))

        completed_part_map[rank] += 1

    return futures
