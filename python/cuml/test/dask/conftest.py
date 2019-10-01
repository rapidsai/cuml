import pytest
from dask_cuda import LocalCUDACluster


@pytest.fixture(scope="module")
def cluster():

    print("Starting local cuda cluster for testing")
    cluster = LocalCUDACluster()
    yield cluster

    print("closing local cuda cluster")
    cluster.close()
