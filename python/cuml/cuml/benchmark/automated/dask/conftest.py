#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


@pytest.fixture(scope="module")
def cluster():
    from dask_cuda import LocalCUDACluster
    from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

    cluster = LocalCUDACluster(
        protocol="tcp",
        scheduler_port=0,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def client(cluster):
    from dask.distributed import Client

    client = Client(cluster)
    yield client
    client.close()


@pytest.fixture(scope="module")
def ucx_cluster():
    from dask_cuda import LocalCUDACluster, initialize
    from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

    initialize.initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
    )
    cluster = LocalCUDACluster(
        protocol="ucx",
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_nvlink=enable_nvlink,
        enable_infiniband=enable_infiniband,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucx_client(ucx_cluster):
    from dask.distributed import Client

    client = Client(ucx_cluster)
    yield client
    client.close()
