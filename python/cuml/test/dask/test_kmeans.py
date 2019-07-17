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
pytestmark = pytest.mark.mg

from dask_cuda import LocalCUDACluster

from cuml.dask.common.comms import default_comms

from dask.distributed import Client, wait


@pytest.mark.skip
def test_end_to_end():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    # NOTE: The LocalCUDACluster needs to be started before any imports that
    # could potentially create a CUDA context.

    import dask_cudf

    import cudf
    import numpy as np

    from cuml.dask.cluster.kmeans import KMeans as cumlKMeans

    def create_df(f, m, n):
        X = np.random.uniform(-1, 1, (m, n))
        ret = cudf.DataFrame([(i,
                               X[:, i].astype(np.float32)) for i in range(n)],
                             index=cudf.dataframe.RangeIndex(f * m,
                                                             f * m + m, 1))
        return ret

    def get_meta(df):
        ret = df.iloc[:0]
        return ret

    def build_dask_df(nrows, ncols):
        workers = client.has_what().keys()

        # Create dfs on each worker (gpu)
        dfs = [client.submit(create_df, n, nrows, ncols, workers=[worker])
               for worker, n in list(zip(workers, list(range(len(workers)))))]
        # Wait for completion
        wait(dfs)
        meta = client.submit(get_meta, dfs[0]).result()
        return dask_cudf.from_delayed(dfs, meta=meta)

    # Per gpu/worker
    train_m = 500
    train_n = 25

    print("Building dask df")
    X_df = build_dask_df(train_m, train_n)

    print("Building model")
    cumlModel = cumlKMeans()

    print("Fitting model")
    cumlModel.fit(X_df)

    print("Predicting model")
    cumlLabels = cumlModel.predict(X_df)

    print(str(cumlLabels.compute()))

    assert False

    default_comms().destroy()
    client.close()
    cluster.close()
