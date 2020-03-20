import pytest
from cuml.dask.datasets.blobs import make_blobs
from cuml.dask.common.input_utils import _extract_partitions
from dask.distributed import Client


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
