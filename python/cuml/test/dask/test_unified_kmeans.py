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
from dask_cuda import LocalCUDACluster

from dask.distributed import Client


@pytest.mark.parametrize('X_datatype', ['cudf', 'dask_cudf'])
@pytest.mark.parametrize('y_datatype', ['cudf', 'dask_cudf'])
@pytest.mark.parametrize('client', ["make", True, None])
@pytest.mark.parametrize('method', ['fit'])
def test_unified_kmeans(X_datatype, y_datatype, client, method):
    if client == 'make':
        cluster = LocalCUDACluster()
        client = Client(cluster)
    if client == True:
        cluster = LocalCUDACluster()
        c = Client(cluster)
        client = True

    import cudf
    import dask_cudf
    from sklearn.datasets import make_regression
    from cuml.cluster.kmeans import KMeans
    from cuml.cluster.sg.kmeans import KMeans as sgKMeans
    from cuml.dask.cluster.kmeans import KMeans as mgKMeans

    X, _, = make_regression(20, 2)
    X, y = X[:16], X[4:][:]
    X = cudf.DataFrame.from_records(X)
    y = cudf.DataFrame.from_records(y)
    if X_datatype == 'dask_cudf':
        X = dask_cudf.from_cudf(X, npartitions=1)
        X.persist()
    if y_datatype == 'dask_cudf':
        y = dask_cudf.from_cudf(y, npartitions=1)
        y.persist()
    try:
        cu_kmeans = KMeans(client=client)
        if method == 'fit':
            cu_kmeans.fit(X)
            cu_kmeans.predict(y)
        elif method == 'fit_predict':
            cu_kmeans.fit_predict(X, y)
    except ValueError:
        if client is None and X_datatype == 'dask_cudf':
            pytest.xfail()
    except TypeError:
        if X_datatype == 'cudf' and y_datatype == 'dask_cudf':
            pytest.xfail()

    if client:
        if client == True:
            c.close()
        else:
            client.close()
        cluster.close()
        assert(isinstance(cu_kmeans.kmeans_obj, mgKMeans))
    else:
        if X_datatype == 'cudf':
            assert(isinstance(cu_kmeans.kmeans_obj, sgKMeans))
        elif X_datatype == 'dask_cudf':
            assert(isinstance(cu_kmeans.kmeans_obj, mgKMeans))