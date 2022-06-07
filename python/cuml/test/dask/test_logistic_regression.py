# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cuml.dask.common import utils as dask_utils
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as skLR
import pandas as pd
import numpy as np
import cupy as cp
import dask_cudf
import cudf

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


def make_classification_dataset(datatype, nrows, ncols, n_info):
    X, y = make_classification(
        n_samples=nrows, n_features=ncols, n_informative=n_info, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


def select_sk_solver(cuml_solver):
    if cuml_solver == 'newton':
        return 'newton-cg'
    elif cuml_solver in ['admm', 'lbfgs']:
        return 'lbfgs'
    else:
        pytest.xfail('No matched sklearn solver')


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e5])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [2, 6])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("gpu_array_input", [False, True])
@pytest.mark.parametrize("solver", ['admm', 'gradient_descent', 'newton',
                                    'lbfgs', 'proximal_grad'])
def test_lr_fit_predict_score(
    nrows, ncols, n_parts, fit_intercept, datatype, gpu_array_input, solver,
    client
):
    sk_solver = select_sk_solver(cuml_solver=solver)

    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    from cuml.dask.extended.linear_model import LogisticRegression \
        as cumlLR_dask

    n_info = 5
    nrows = np.int(nrows)
    ncols = np.int(ncols)
    X, y = make_classification_dataset(datatype, nrows, ncols, n_info)

    gX, gy = _prep_training_data(client, X, y, n_parts)

    if gpu_array_input:
        gX = gX.values
        gX._meta = cp.asarray(gX._meta)
        gy = gy.values
        gy._meta = cp.asarray(gy._meta)

    cuml_model = cumlLR_dask(fit_intercept=fit_intercept,
                             solver=solver,
                             max_iter=10)

    # test fit and predict
    cuml_model.fit(gX, gy)
    cu_preds = cuml_model.predict(gX)
    accuracy_cuml = accuracy_score(y, cu_preds.compute().get())

    sk_model = skLR(fit_intercept=fit_intercept, solver=sk_solver, max_iter=10)
    sk_model.fit(X, y)
    sk_preds = sk_model.predict(X)
    accuracy_sk = accuracy_score(y, sk_preds)

    assert (accuracy_cuml >= accuracy_sk) | \
        (np.abs(accuracy_cuml - accuracy_sk) < 1e-3)

    # score
    accuracy_cuml = cuml_model.score(gX, gy).compute().item()
    accuracy_sk = sk_model.score(X, y)

    assert (accuracy_cuml >= accuracy_sk) | \
        (np.abs(accuracy_cuml - accuracy_sk) < 1e-3)

    # predicted probabilities should differ by <= 5%
    # even with different solvers (arbitrary)
    probs_cuml = cuml_model.predict_proba(gX).compute()
    probs_sk = sk_model.predict_proba(X)[:, 1]
    assert np.abs(probs_sk - probs_cuml.get()).max() <= 0.05
