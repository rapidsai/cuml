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

from dask_cuda import LocalCUDACluster

from dask.distributed import Client, wait

from cuml.dask.common import CommsBase
from cuml.dask.common import perform_test_comms_send_recv
from cuml.dask.common import perform_test_comms_allreduce


def test_allreduce():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    cb = CommsBase(comms_p2p = True, comms_coll = True)
    cb.init()

    workers = client.has_what().keys()

    print(str(workers))

    # Create dfs on each worker (gpu)
    dfs = [client.submit(perform_test_comms_allreduce, handle, workers = [w])
           for wid, w, handle in cb.handles]


    # Wait for completion
    wait(dfs)

    print(str(list(map(lambda x: x.result(), dfs))))

    assert all(list(map(lambda x: x.result(), dfs)))


def test_send_recv():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    cb = CommsBase(comms_p2p=True, comms_coll=True)
    cb.init()

    workers = client.has_what().keys()

    print(str(workers))

    dfs = [client.submit(perform_test_comms_send_recv, handle, workers=[w])
           for wid, w, handle in cb.handles]

    wait(dfs)

    print(str(list(map(lambda x: x.result(), dfs))))

    assert(list(map(lambda x: x.result(), dfs)))
