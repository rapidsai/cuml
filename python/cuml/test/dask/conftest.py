import pytest
from dask_cuda import LocalCUDACluster

from dask.distributed import Client, wait


@pytest.fixture(scope="module")
def cluster():

    print("Starting local cuda cluster for testing")
    cluster = LocalCUDACluster()
    yield cluster

    print("closing local cuda cluster")
    cluster.close()