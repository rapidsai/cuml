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

from collections.abc import Iterable

from tornado import gen
from dask.distributed import default_client
from toolz import first

from dask.distributed import wait
from dask import delayed


@gen.coroutine
def extract_arr_partitions(arrs, client=None):
    """
    Given a Dask Array, return an array of tuples mapping each
    worker to their list of futures.

    :param arr: Dask.array split array partitions into a list of
               futures.
    :param client: dask.distributed.Client Optional client to use
    """
    client = default_client() if client is None else client

    if not isinstance(arrs, Iterable):
        dist_arr = arrs.to_delayed().ravel()
        to_map = dist_arr
    else:
        parts = [arr.to_delayed().ravel() for arr in arrs]
        to_map = zip(*parts)

    parts = list(map(delayed, to_map))
    parts = client.compute(parts)

    yield wait(parts)

    who_has = yield client.who_has(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])

    worker_map = {}  # Map from part -> worker
    for key, workers in who_has.items():
        worker = first(workers)
        worker_map[key_to_part_dict[key]] = worker

    worker_to_parts = []
    for part in parts:
        worker = worker_map[part]
        worker_to_parts.append((worker, part))

    yield wait(worker_to_parts)
    raise gen.Return(worker_to_parts)
