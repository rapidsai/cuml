import pytest
from cuml.dask.datasets.blobs import make_blobs
from cuml.dask.common.part_utils import _extract_partitions
from dask.distributed import Client
import dask.array as da
import cupy as cp


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e4])
@pytest.mark.parametrize("ncols", [10])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
@pytest.mark.parametrize("colocated", [True, False])
def test_extract_partitions_worker_list(nrows, ncols, n_parts, input_type,
                                        colocated, cluster):
    client = Client(cluster)

    try:
        X, y = make_blobs(nrows=nrows, ncols=ncols, n_parts=n_parts,
                          output=input_type)

        if colocated:
            gpu_futures = client.sync(_extract_partitions, (X, y), client)
        else:
            gpu_futures = client.sync(_extract_partitions, X, client)

        parts = list(map(lambda x: x[1], gpu_futures))
        assert len(parts) == n_parts
    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [24])
@pytest.mark.parametrize("ncols", [2])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
@pytest.mark.parametrize("colocated", [True, False])
def test_extract_partitions_shape(nrows, ncols, n_parts, input_type,
                                  colocated, cluster):
    client = Client(cluster)

    try:
        X, y = make_blobs(nrows=nrows, ncols=ncols, n_parts=n_parts,
                          output=input_type)
        if input_type == "dataframe":
            X_len_parts = X.map_partitions(len).compute()
            y_len_parts = y.map_partitions(len).compute()
        elif input_type == "array":
            X_len_parts = X.chunks[0]
            y_len_parts = y.chunks[0]

        if colocated:
            gpu_futures = client.sync(_extract_partitions, (X, y), client)
        else:
            gpu_futures = client.sync(_extract_partitions, X, client)

        parts = [part.result() for worker, part in gpu_futures]

        if colocated:
            for i in range(len(parts)):
                assert (parts[i][0].shape[0] == X_len_parts[i]) and (
                        parts[i][1].shape[0] == y_len_parts[i])
        else:
            for i in range(len(parts)):
                assert (parts[i].shape[0] == X_len_parts[i])

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [24])
@pytest.mark.parametrize("ncols", [2])
@pytest.mark.parametrize("n_parts", [2, 12])
@pytest.mark.parametrize("X_delayed", [True, False])
@pytest.mark.parametrize("y_delayed", [True, False])
@pytest.mark.parametrize("colocated", [True, False])
def test_extract_partitions_futures(nrows, ncols, n_parts, X_delayed,
                                    y_delayed, colocated, cluster):

    client = Client(cluster)
    try:

        X = cp.random.standard_normal((nrows, ncols))
        y = cp.random.standard_normal((nrows, ))

        X = da.from_array(X, chunks=(nrows/n_parts, -1))
        y = da.from_array(y, chunks=(nrows/n_parts, ))

        if not X_delayed:
            X = client.persist(X)
        if not y_delayed:
            y = client.persist(y)

        if colocated:
            gpu_futures = client.sync(_extract_partitions, (X, y), client)
        else:
            gpu_futures = client.sync(_extract_partitions, X, client)

        parts = list(map(lambda x: x[1], gpu_futures))
        assert len(parts) == n_parts

    finally:
        client.close()
