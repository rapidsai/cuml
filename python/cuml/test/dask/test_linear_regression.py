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

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from sklearn.metrics import mean_squared_error
import pandas as pd
import gzip
import numpy as np
import os

pytestmark = pytest.mark.mg


def load_data(nrows, ncols, cached='data/mortgage.npy.gz'):
    # Loading into pandas to not create any clusters before LocalCUDACluster
    if os.path.exists(cached):
        print('use mortgage data')
        with gzip.open(cached) as f:
            X = np.load(f)
        # the 4th column is 'adj_remaining_months_to_maturity'
        # used as the label
        X = X[:, [i for i in range(X.shape[1]) if i != 4]]
        y = X[:, 4:5]
        rindices = np.random.randint(0, X.shape[0]-1, nrows)
        X = X[rindices, :ncols]
        y = y[rindices]
    else:
        print('use random data')
        X = np.random.rand(nrows, ncols)

    df_X = pd.DataFrame({'fea%d' % i: X[:, i] for i in range(X.shape[1])})
    df_y = pd.DataFrame({'fea%d' % i: y[:, i] for i in range(y.shape[1])})

    return df_X, df_y


@pytest.mark.skip(reason="Test should be run only with libcuML.so")
def test_ols():

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    import dask_cudf

    import cudf
    import numpy as np

    from cuml.dask.linear_model import LinearRegression as cumlOLS_dask

    nrows = 2**8
    ncols = 399

    X, y = load_data(nrows, ncols)

    X_cudf = cudf.DataFrame.from_pandas(X)
    y_cudf = np.array(y.as_matrix())
    y_cudf = y_cudf[:, 0]
    y_cudf = cudf.Series(y_cudf)

    workers = client.has_what().keys()

    X_df = dask_cudf.from_cudf(X_cudf, npartitions=len(workers)).persist()
    y_df = dask_cudf.from_cudf(y_cudf, npartitions=len(workers)).persist()

    lr = cumlOLS_dask()

    lr.fit(X_df, y_df)

    ret = lr.predict(X_df)

    error_cuml = mean_squared_error(y, ret.compute().to_array())

    assert(error_cuml < 1e-6)
