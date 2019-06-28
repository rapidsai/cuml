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

import pytest
from dask_cuda import LocalCUDACluster

from dask.distributed import Client, wait

from cuml.dask.common import CommsBase
from cuml.dask.common import perform_test_comms_send_recv
from cuml.dask.common import perform_test_comms_allreduce


@pytest.mark.skip
def test_comms_init_no_p2p():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)  # noqa

    cb = CommsBase()
    cb.init()

    assert cb.nccl_initialized is True
    assert cb.ucx_initialized is False

    cb = CommsBase(comms_p2p=True)
    cb.init()

    assert cb.nccl_initialized is True
    assert cb.ucx_initialized is True


@pytest.mark.skip
def test_allreduce():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    cb = CommsBase()
    cb.init()

    workers = client.has_what().keys()

    print(str(workers))

    dfs = [client.submit(perform_test_comms_allreduce, handle, workers=[w])
           for wid, w, handle in cb.handles]

    wait(dfs)

    print(str(list(map(lambda x: x.result(), dfs))))

    assert all(list(map(lambda x: x.result(), dfs)))

    # todo: Destroy is failing here. Need to fix it
    # cb.destroy()


@pytest.mark.skip
def test_send_recv(n_trials):

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    cb = CommsBase(comms_p2p=True)
    cb.init()

    workers = client.has_what().keys()

    print(str(workers))

    dfs = [client.submit(perform_test_comms_send_recv, handle,
                         n_trials, workers=[w])
           for wid, w, handle in cb.handles]

    wait(dfs)

    print(str(list(map(lambda x: x.result(), dfs))))

    assert(list(map(lambda x: x.result(), dfs)))

    # todo: Destroy is failing here. Need to fix it
    # cb.destroy()
