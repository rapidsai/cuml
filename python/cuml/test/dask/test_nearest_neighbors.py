# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest

import cudf
import dask_cudf
import pandas as pd

from cuml.dask.common import utils as dask_utils

from dask.distributed import Client, wait

from cuml.test.utils import unit_param, quality_param, stress_param


def _prep_training_data(c, X_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    X_train_df, = dask_utils.persist_across_workers(c,
                                                    [X_train_df],
                                                    workers=workers)
    return X_train_df


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(1e3), quality_param(1e5),
                                   stress_param(5e6)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
def test_end_to_end(nrows, ncols, nclusters, n_parts, cluster):

    client = Client(cluster)

    try:
        from cuml.dask.neighbors import NearestNeighbors as cumlNN

        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=int(nrows), n_features=ncols,
                          centers=nclusters, cluster_std=0.01)

        X_cudf = _prep_training_data(client, X, 5)

        wait(X_cudf)

        cumlModel = cumlNN(verbose=1, n_neighbors=10)

        cumlModel.fit(X_cudf)

        out_d, out_i = cumlModel.kneighbors(X_cudf)

        print(str(out_i.compute()))

        from sklearn.neighbors import NearestNeighbors

        print(str(NearestNeighbors().fit(X).kneighbors(X, 10)))

    finally:
        client.close()
