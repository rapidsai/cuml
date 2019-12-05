import pytest

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster

enable_tcp_over_ucx = True


@pytest.fixture(scope="module")
def cluster():
    cluster = LocalCUDACluster()
    yield cluster
    cluster.close()


@pytest.fixture(scope="module")
def ucx_cluster():
    initialize.initialize(create_cuda_context=True,
                          enable_tcp_over_ucx=enable_tcp_over_ucx)
    cluster = LocalCUDACluster(protocol="ucx")
    yield cluster
    cluster.close()
