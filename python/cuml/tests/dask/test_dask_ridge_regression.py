# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import gpu_only_import
import pytest
from cuml.dask.common import utils as dask_utils
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from cuml.internals.safe_imports import cpu_only_import

pd = cpu_only_import("pandas")
np = cpu_only_import("numpy")
dask_cudf = gpu_only_import("dask_cudf")
cudf = gpu_only_import("cudf")

pytestmark = pytest.mark.mg


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = np.array(pd.DataFrame(y_train).values)
    y_cudf = y_cudf[:, 0]
    y_cudf = cudf.Series(y_cudf)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


def make_regression_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=5, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e4])
@pytest.mark.parametrize("ncols", [10])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("normalize", [False])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("delayed", [True, False])
def test_ridge(
    nrows, ncols, n_parts, fit_intercept, normalize, datatype, delayed, client
):

    from cuml.dask.linear_model import Ridge as cumlRidge_dask

    n_info = 5
    nrows = int(nrows)
    ncols = int(ncols)
    X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    lr = cumlRidge_dask(
        alpha=0.5, fit_intercept=fit_intercept, normalize=normalize
    )

    lr.fit(X_df, y_df)

    ret = lr.predict(X_df, delayed=delayed)

    error_cuml = mean_squared_error(y, ret.compute().to_pandas().values)

    assert error_cuml < 1e-1
