import pytest

import dask

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster

enable_tcp_over_ucx = True
enable_nvlink = True
enable_infiniband = True


@pytest.fixture(scope="module")
def cluster():

    dask.config.set({"distributed.comm.timeouts.connect": "50s"})

    cluster = LocalCUDACluster(protocol="tcp", threads_per_worker=25)
    yield cluster
    cluster.close()


@pytest.fixture(scope="module")
def ucx_cluster():
    initialize.initialize(create_cuda_context=True,
                         
                          enable_tcp_over_ucx=enable_tcp_over_ucx,
                          enable_nvlink=enable_nvlink,
                          enable_infiniband=enable_infiniband)
    cluster = LocalCUDACluster(protocol="ucx",
                               interface="enp1s0f0",
                               threads_per_worker=5,
                               enable_tcp_over_ucx=enable_tcp_over_ucx,
                               enable_nvlink=enable_nvlink,
                               enable_infiniband=enable_infiniband,
                               ucx_net_devices="enp1s0f0")
    yield cluster
    cluster.close()
