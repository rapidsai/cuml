# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import pytest

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny
from dask.distributed import Client

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


@pytest.fixture(scope="module")
def cluster():

    cluster = LocalCUDACluster(
        protocol="tcp",
        scheduler_port=0,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def client(cluster):

    client = Client(cluster)
    yield client
    client.close()


@pytest.fixture(scope="module")
def ucx_cluster():
    cluster = LocalCUDACluster(
        protocol="ucx",
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucx_client(ucx_cluster):

    client = Client(ucx_cluster)
    yield client
    client.close()


@pytest.fixture(scope="module")
def ucxx_cluster():
    cluster = LocalCUDACluster(
        protocol="ucxx",
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucxx_client(ucxx_cluster):
    pytest.importorskip("distributed_ucxx")

    client = Client(ucxx_cluster)
    yield client
    client.close()
