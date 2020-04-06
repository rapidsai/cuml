import pytest

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


@pytest.fixture(scope="module")
def cluster():

    print("Starting cluster")
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    yield cluster
    print("Closing cluster")
    cluster.close()
    print("Closed cluster")


@pytest.fixture(scope="module")
def ucx_cluster():
    initialize.initialize(create_cuda_context=True,
                          enable_tcp_over_ucx=enable_tcp_over_ucx,
                          enable_nvlink=enable_nvlink,
                          enable_infiniband=enable_infiniband)
    cluster = LocalCUDACluster(protocol="ucx",
                               enable_tcp_over_ucx=enable_tcp_over_ucx,
                               enable_nvlink=enable_nvlink,
                               enable_infiniband=enable_infiniband,
                               ucx_net_devices="auto")
    yield cluster
    cluster.close()
