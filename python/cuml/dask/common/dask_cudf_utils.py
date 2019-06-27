from tornado import gen
from dask.distributed import default_client
from toolz import first
import logging
import dask.dataframe as dd

import dask_cudf
import numpy as np
import cudf
import pandas as pd

from dask.distributed import wait


@gen.coroutine
def extract_ddf_partitions(ddf):
    """
    Given a Dask cuDF, return a tuple with (worker, future) for each partition
    """
    client = default_client()

    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    yield wait(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = [(worker, part) for worker, part in worker_map]

    yield wait(gpu_data)

    raise gen.Return(gpu_data)


def get_meta(df):
    ret = df.iloc[:0]
    return ret


def to_dask_cudf(futures):
    # Convert a list of futures containing dfs back into a dask_cudf
    dfs = [d for d in futures if d.type != type(None)]
    meta = c.submit(get_meta, dfs[0]).result()
    return dd.from_delayed(dfs, meta=meta)
