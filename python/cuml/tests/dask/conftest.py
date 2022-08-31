# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import pytest

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


@pytest.fixture(scope="module")
def cluster():

    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def client(cluster):

    client = Client(cluster)
    yield client
    client.close()


@pytest.fixture(scope="module")
def ucx_cluster():
    initialize.initialize(create_cuda_context=True,
                          enable_tcp_over_ucx=enable_tcp_over_ucx,
                          enable_nvlink=enable_nvlink,
                          enable_infiniband=enable_infiniband)
    cluster = LocalCUDACluster(protocol="ucx",
                               enable_tcp_over_ucx=enable_tcp_over_ucx,
                               enable_nvlink=enable_nvlink,
                               enable_infiniband=enable_infiniband)
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucx_client(ucx_cluster):

    client = Client(ucx_cluster)
    yield client
    client.close()
