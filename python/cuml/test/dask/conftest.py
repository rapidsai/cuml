import pytest

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster

@pytest.fixture(scope="module")
def cluster():
    initialize.initialize(create_cuda_context=True,
                          enable_tcp_over_ucx=True)
    cluster = LocalCUDACluster(protocol="ucx", threads_per_worker=1)
    yield cluster
    cluster.close()
