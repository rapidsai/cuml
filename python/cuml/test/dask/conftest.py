import pytest
from dask_cuda import LocalCUDACluster


@pytest.fixture(scope="module")
def cluster():
    cluster = LocalCUDACluster()
    yield cluster
    cluster.close()
