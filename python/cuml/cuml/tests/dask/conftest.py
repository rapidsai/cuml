# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from ssl import create_default_context
from urllib.request import HTTPSHandler, build_opener, install_opener

import certifi
import pytest

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


# Install SSL certificates
def pytest_sessionstart(session):
    ssl_context = create_default_context(cafile=certifi.where())
    https_handler = HTTPSHandler(context=ssl_context)
    install_opener(build_opener(https_handler))


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
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(
        protocol="ucx",
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucx_client(ucx_cluster):
    from dask.distributed import Client

    client = Client(ucx_cluster)
    yield client
    client.close()


@pytest.fixture(scope="module")
def ucxx_cluster():
    from dask_cuda import LocalCUDACluster
    from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

    cluster = LocalCUDACluster(
        protocol="ucxx",
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def ucxx_client(ucxx_cluster):
    pytest.importorskip("distributed_ucxx")
    from dask.distributed import Client

    client = Client(ucxx_cluster)
    yield client
    client.close()


def pytest_addoption(parser):
    group = parser.getgroup("Dask cuML Custom Options")

    group.addoption(
        "--run_ucx",
        action="store_true",
        default=False,
        help="run _only_ UCX-Py tests",
    )

    group.addoption(
        "--run_ucxx",
        action="store_true",
        default=False,
        help="run _only_ UCXX tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_ucx"):
        skip_others = pytest.mark.skip(
            reason="only runs when --run_ucx is not specified"
        )
        for item in items:
            if "ucx" not in item.keywords:
                item.add_marker(skip_others)
    else:
        skip_ucx = pytest.mark.skip(reason="requires --run_ucx to run")
        for item in items:
            if "ucx" in item.keywords:
                item.add_marker(skip_ucx)

    if config.getoption("--run_ucxx"):
        skip_others = pytest.mark.skip(
            reason="only runs when --run_ucxx is not specified"
        )
        for item in items:
            if "ucxx" not in item.keywords:
                item.add_marker(skip_others)
    else:
        skip_ucxx = pytest.mark.skip(reason="requires --run_ucxx to run")
        for item in items:
            if "ucxx" in item.keywords:
                item.add_marker(skip_ucxx)
