
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import dask_cudf
import pytest

import numpy as np
import pandas as pd

from cuml.dask.ensemble import RandomForestClassifier as cuRFC_mg
from cuml.dask.ensemble import RandomForestRegressor as cuRFR_mg
from cuml.dask.common import utils as dask_utils

from dask.array import from_array
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier as skrfc


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = \
        dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, \
        y_train_df = dask_utils.persist_across_workers(c,
                                                       [X_train_df,
                                                        y_train_df],
                                                       workers=workers)
    return X_train_df, y_train_df


@pytest.mark.parametrize('partitions_per_worker', [3])
def test_rf_classification_dask_cudf(partitions_per_worker, client):

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    X, y = make_classification(n_samples=10000, n_features=20,
                               n_clusters_per_class=1, n_informative=10,
                               random_state=123, n_classes=5)

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=1000)

    cu_rf_params = {
        'n_estimators': 40,
        'max_depth': 16,
        'n_bins': 16,
    }

    X_train_df, y_train_df = _prep_training_data(client, X_train, y_train,
                                                 partitions_per_worker)

    cuml_mod = cuRFC_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)
    cuml_mod_predict = cuml_mod.predict(X_test)
    acc_score = accuracy_score(cuml_mod_predict, y_test, normalize=True)

    assert acc_score > 0.8


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('partitions_per_worker', [5])
def test_rf_regression_dask_fil(partitions_per_worker,
                                dtype, client):
    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    X, y = make_regression(n_samples=10000, n_features=20,
                           n_informative=10, random_state=123)

    X = X.astype(dtype)
    y = y.astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=1000,
                                                        random_state=123)

    if dtype == np.float64:
        pytest.xfail(reason=" Dask RF does not support np.float64 data")

    cu_rf_params = {
        'n_estimators': 50,
        'max_depth': 16,
        'n_bins': 16,
    }

    workers = client.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)

    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = \
        dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = \
        dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)
    X_cudf_test = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
    X_test_df = \
        dask_cudf.from_cudf(X_cudf_test, npartitions=n_partitions)

    cuml_mod = cuRFR_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)

    cuml_mod_predict = cuml_mod.predict(X_test_df)
    cuml_mod_predict = cp.asnumpy(cp.array(cuml_mod_predict.compute()))

    acc_score = r2_score(cuml_mod_predict, y_test)

    assert acc_score >= 0.67


@pytest.mark.parametrize('partitions_per_worker', [5])
@pytest.mark.parametrize('output_class', [True, False])
def test_rf_classification_dask_array(partitions_per_worker, client,
                                      output_class):

    X, y = make_classification(n_samples=10000, n_features=30,
                               n_clusters_per_class=1, n_informative=20,
                               random_state=123, n_classes=2)

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=1000)

    cu_rf_params = {
        'n_estimators': 25,
        'max_depth': 13,
        'n_bins': 15,
    }

    X_train_df, y_train_df = _prep_training_data(client, X_train, y_train,
                                                 partitions_per_worker)
    X_test_dask_array = from_array(X_test)
    cuml_mod = cuRFC_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)
    cuml_mod_predict = cuml_mod.predict(X_test_dask_array,
                                        output_class).compute()
    if not output_class:
        cuml_mod_predict = np.round(cuml_mod_predict)

    acc_score = accuracy_score(cuml_mod_predict, y_test, normalize=True)

    assert acc_score > 0.8


@pytest.mark.parametrize('partitions_per_worker', [5])
def test_rf_regression_dask_cpu(partitions_per_worker, client):
    X, y = make_regression(n_samples=10000, n_features=20,
                           n_informative=10, random_state=123)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=1000,
                                                        random_state=123)

    cu_rf_params = {
        'n_estimators': 50,
        'max_depth': 16,
        'n_bins': 16,
    }

    workers = client.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)

    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = \
        dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = \
        dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        client, [X_train_df, y_train_df], workers=workers)

    cuml_mod = cuRFR_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)

    cuml_mod_predict = cuml_mod.predict(X_test, predict_model='CPU')

    acc_score = r2_score(cuml_mod_predict, y_test)

    assert acc_score >= 0.67


@pytest.mark.parametrize('partitions_per_worker', [5])
def test_rf_classification_dask_fil_predict_proba(partitions_per_worker,
                                                  client):
    X, y = make_classification(n_samples=1000, n_features=30,
                               n_clusters_per_class=1, n_informative=20,
                               random_state=123, n_classes=2)

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=100, random_state=123)

    cu_rf_params = {'n_bins': 16, 'n_streams': 1,
                    'n_estimators': 40, 'max_depth': 16
                    }

    X_train_df, y_train_df = _prep_training_data(client, X_train, y_train,
                                                 partitions_per_worker)
    X_test_df, _ = _prep_training_data(client, X_test, y_test,
                                       partitions_per_worker)
    cu_rf_mg = cuRFC_mg(**cu_rf_params)
    cu_rf_mg.fit(X_train_df, y_train_df)

    fil_preds_proba = cu_rf_mg.predict_proba(X_test_df).compute()
    fil_preds_proba = cp.asnumpy(fil_preds_proba.to_gpu_matrix())
    y_proba = np.zeros(np.shape(fil_preds_proba))
    y_proba[:, 1] = y_test
    y_proba[:, 0] = 1.0 - y_test
    fil_mse = mean_squared_error(y_proba, fil_preds_proba)
    sk_model = skrfc(n_estimators=40,
                     max_depth=16,
                     random_state=10)
    sk_model.fit(X_train, y_train)
    sk_preds_proba = sk_model.predict_proba(X_test)
    sk_mse = mean_squared_error(y_proba, sk_preds_proba)

    # The threshold is required as the test would intermitently
    # fail with a max difference of 0.022 between the two mse values
    assert fil_mse <= sk_mse + 0.022


@pytest.mark.parametrize('model_type', ['classification', 'regression'])
def test_rf_concatenation_dask(client, model_type):
    from cuml.fil.fil import TreeliteModel
    X, y = make_classification(n_samples=1000, n_features=30,
                               random_state=123, n_classes=2)

    X = X.astype(np.float32)
    if model_type == 'classification':
        y = y.astype(np.int32)
    else:
        y = y.astype(np.float32)
    n_estimators = 40
    cu_rf_params = {'n_estimators': n_estimators}

    X_df, y_df = _prep_training_data(client, X, y,
                                     partitions_per_worker=2)

    if model_type == 'classification':
        cu_rf_mg = cuRFC_mg(**cu_rf_params)
    else:
        cu_rf_mg = cuRFR_mg(**cu_rf_params)

    cu_rf_mg.fit(X_df, y_df)
    res1 = cu_rf_mg.predict(X_df)
    res1.compute()
    local_tl = TreeliteModel.from_treelite_model_handle(
        cu_rf_mg.internal_model._obtain_treelite_handle(),
        take_handle_ownership=False)

    assert local_tl.num_trees == n_estimators
