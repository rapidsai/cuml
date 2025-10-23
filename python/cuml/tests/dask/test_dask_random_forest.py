# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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


# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import json

import cudf
import cupy as cp
import dask_cudf
import numpy as np
import pandas as pd
import pytest
import treelite
from dask.array import from_array
from dask.distributed import Client
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from cuml.dask._compat import DASK_2025_4_0
from cuml.dask.common import utils as dask_utils
from cuml.dask.ensemble import RandomForestClassifier as cuRFC_mg
from cuml.dask.ensemble import RandomForestRegressor as cuRFR_mg
from cuml.ensemble import RandomForestClassifier as cuRFC_sg
from cuml.ensemble import RandomForestRegressor as cuRFR_sg


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


@pytest.mark.parametrize("partitions_per_worker", [3])
def test_rf_classification_multi_class(partitions_per_worker, cluster):

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    c = Client(cluster)
    kwargs = {"n_workers": -1} if DASK_2025_4_0() else {}
    n_workers = len(c.scheduler_info(**kwargs)["workers"])

    try:

        X, y = make_classification(
            n_samples=n_workers * 5000,
            n_features=20,
            n_clusters_per_class=1,
            n_informative=10,
            random_state=123,
            n_classes=15,
        )

        X = X.astype(np.float32)
        y = y.astype(np.int32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_workers * 300, random_state=123
        )

        cu_rf_params = {
            "n_estimators": n_workers * 8,
            "max_depth": 16,
            "n_bins": 256,
            "random_state": 10,
        }

        X_train_df, y_train_df = _prep_training_data(
            c, X_train, y_train, partitions_per_worker
        )

        cuml_mod = cuRFC_mg(**cu_rf_params, ignore_empty_partitions=True)
        cuml_mod.fit(X_train_df, y_train_df)
        X_test_dask_array = from_array(X_test)
        cuml_preds_gpu = cuml_mod.predict(X_test_dask_array).compute()
        acc_score_gpu = accuracy_score(cuml_preds_gpu, y_test)

        # the sklearn model when ran with the same parameters gives an
        # accuracy of 0.69. There is a difference of 0.0632 (6.32%) between
        # the two when the code runs on a single GPU (seen in the CI)
        # Refer to issue : https://github.com/rapidsai/cuml/issues/2806 for
        # more information on the threshold value.

        assert acc_score_gpu >= 0.52

    finally:
        c.close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_regression_dask_fil(partitions_per_worker, dtype, client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    X, y = make_regression(
        n_samples=n_workers * 4000,
        n_features=20,
        n_informative=10,
        random_state=123,
    )

    X = X.astype(dtype)
    y = y.astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 100, random_state=123
    )

    cu_rf_params = {
        "n_estimators": 50,
        "max_depth": 16,
        "n_bins": 16,
    }

    workers = client.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)

    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)
    X_cudf_test = cudf.DataFrame(pd.DataFrame(X_test))
    X_test_df = dask_cudf.from_cudf(X_cudf_test, npartitions=n_partitions)

    cuml_mod = cuRFR_mg(**cu_rf_params, ignore_empty_partitions=True)
    cuml_mod.fit(X_train_df, y_train_df)

    cuml_mod_predict = cuml_mod.predict(X_test_df)
    cuml_mod_predict = cp.asnumpy(cp.array(cuml_mod_predict.compute()))

    acc_score = r2_score(y_test, cuml_mod_predict)

    assert acc_score >= 0.59


@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_classification_dask_array(partitions_per_worker, client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 2000,
        n_features=30,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=123,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 400
    )

    cu_rf_params = {
        "n_estimators": 25,
        "max_depth": 13,
        "n_bins": 15,
    }

    X_train_df, y_train_df = _prep_training_data(
        client, X_train, y_train, partitions_per_worker
    )
    X_test_dask_array = from_array(X_test)
    cuml_mod = cuRFC_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)
    cuml_mod_predict = cuml_mod.predict(X_test_dask_array).compute()

    acc_score = accuracy_score(cuml_mod_predict, y_test, normalize=True)

    assert acc_score > 0.8


@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_classification_dask_fil_predict_proba(
    partitions_per_worker, client
):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 1500,
        n_features=30,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=123,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 150, random_state=123
    )

    cu_rf_params = {
        "n_bins": 16,
        "n_streams": 1,
        "n_estimators": 40,
        "max_depth": 16,
    }

    X_train_df, y_train_df = _prep_training_data(
        client, X_train, y_train, partitions_per_worker
    )
    X_test_df, _ = _prep_training_data(
        client, X_test, y_test, partitions_per_worker
    )
    cu_rf_mg = cuRFC_mg(**cu_rf_params)
    cu_rf_mg.fit(X_train_df, y_train_df)

    fil_preds = cu_rf_mg.predict(X_test_df).compute()
    fil_preds = fil_preds.to_numpy()
    fil_preds_proba = cu_rf_mg.predict_proba(X_test_df).compute()
    fil_preds_proba = fil_preds_proba.to_numpy()
    np.testing.assert_equal(fil_preds, np.argmax(fil_preds_proba, axis=1))

    y_proba = np.zeros(np.shape(fil_preds_proba))
    y_proba[:, 1] = y_test
    y_proba[:, 0] = 1.0 - y_test
    fil_mse = mean_squared_error(y_proba, fil_preds_proba)
    sk_model = skrfc(n_estimators=40, max_depth=16, random_state=10)
    sk_model.fit(X_train, y_train)
    sk_preds_proba = sk_model.predict_proba(X_test)
    sk_mse = mean_squared_error(y_proba, sk_preds_proba)

    # The threshold is required as the test would intermitently
    # fail with a max difference of 0.029 between the two mse values
    assert fil_mse <= sk_mse + 0.029


@pytest.mark.parametrize("model_type", ["classification", "regression"])
def test_rf_concatenation_dask(client, model_type):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 200, n_features=30, random_state=123, n_classes=2
    )

    X = X.astype(np.float32)
    if model_type == "classification":
        y = y.astype(np.int32)
    else:
        y = y.astype(np.float32)
    n_estimators = 40
    cu_rf_params = {"n_estimators": n_estimators}

    X_df, y_df = _prep_training_data(client, X, y, partitions_per_worker=2)

    if model_type == "classification":
        cu_rf_mg = cuRFC_mg(**cu_rf_params)
    else:
        cu_rf_mg = cuRFR_mg(**cu_rf_params)

    cu_rf_mg.fit(X_df, y_df)
    res1 = cu_rf_mg.predict(X_df)
    res1.compute()
    if cu_rf_mg.internal_model:
        treelite_bytes = cu_rf_mg.internal_model._treelite_model_bytes
        local_tl = treelite.Model.deserialize_bytes(treelite_bytes)
        assert local_tl.num_tree == n_estimators


@pytest.mark.parametrize("ignore_empty_partitions", [True, False])
def test_single_input_regression(client, ignore_empty_partitions):
    X, y = make_classification(n_samples=1, n_classes=1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X, y = _prep_training_data(client, X, y, partitions_per_worker=2)
    cu_rf_mg = cuRFR_mg(
        n_bins=1, ignore_empty_partitions=ignore_empty_partitions
    )

    if (
        ignore_empty_partitions
        or len(client.scheduler_info(n_workers=-1)["workers"].keys()) == 1
    ):
        cu_rf_mg.fit(X, y)
        cuml_mod_predict = cu_rf_mg.predict(X)
        cuml_mod_predict = cp.asnumpy(cp.array(cuml_mod_predict.compute()))
        y = cp.asnumpy(cp.array(y.compute()))
        assert y[0] == cuml_mod_predict[0]

    else:
        with pytest.raises(ValueError):
            cu_rf_mg.fit(X, y)


@pytest.mark.parametrize("max_depth", [1, 2, 3, 5, 10, 15, 20])
@pytest.mark.parametrize("n_estimators", [5, 10, 20])
def test_rf_data_count(client, max_depth, n_estimators):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    if n_estimators < n_workers:
        err_msg = "n_estimators cannot be lower than number of dask workers"
        pytest.xfail(err_msg)

    n_samples_per_worker = 350

    X, y = make_classification(
        n_samples=n_samples_per_worker * n_workers,
        n_features=20,
        n_clusters_per_class=1,
        n_informative=10,
        random_state=123,
        n_classes=2,
    )
    X = X.astype(np.float32)
    dask_model = cuRFC_mg(
        max_features=1.0,
        max_samples=1.0,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=23707,
        n_streams=1,
        n_estimators=n_estimators,
        max_leaves=-1,
        max_depth=max_depth,
    )
    y = y.astype(np.int32)

    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=2)
    dask_model.fit(X_dask, y_dask)
    model = dask_model.get_combined_model()
    json_obj = json.loads(model.as_treelite().dump_as_json())

    def check_count(node, nodes):
        if "left_child" in node:
            left = nodes[node["left_child"]]
            right = nodes[node["right_child"]]
            count = check_count(left, nodes) + check_count(right, nodes)
            assert count == node["data_count"]
        return node["data_count"]

    for tree in json_obj["trees"]:
        nodes = tree["nodes"]
        # The root's count should be equal to the number of rows in the data
        assert nodes[0]["data_count"] == n_samples_per_worker
        # Check that the data_count accumulates properly as you move up the tree
        for node in nodes:
            check_count(node, nodes)


@pytest.mark.parametrize("estimator_type", ["regression", "classification"])
def test_rf_get_combined_model_right_aftter_fit(client, estimator_type):
    max_depth = 3
    n_estimators = 5

    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    if n_estimators < n_workers:
        err_msg = "n_estimators cannot be lower than number of dask workers"
        pytest.xfail(err_msg)

    X, y = make_classification()
    X = X.astype(np.float32)
    if estimator_type == "classification":
        cu_rf_mg = cuRFC_mg(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            n_streams=1,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.int32)
    elif estimator_type == "regression":
        cu_rf_mg = cuRFR_mg(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            n_streams=1,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.float32)
    else:
        assert False
    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=2)
    cu_rf_mg.fit(X_dask, y_dask)
    single_gpu_model = cu_rf_mg.get_combined_model()
    if estimator_type == "classification":
        assert isinstance(single_gpu_model, cuRFC_sg)
    elif estimator_type == "regression":
        assert isinstance(single_gpu_model, cuRFR_sg)
    else:
        assert False


@pytest.mark.parametrize("model_type", ["classification", "regression"])
@pytest.mark.parametrize("fit_broadcast", [True, False])
@pytest.mark.parametrize("transform_broadcast", [True, False])
def test_rf_broadcast(model_type, fit_broadcast, transform_broadcast, client):
    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    workers = list(client.scheduler_info(n_workers=-1)["workers"].keys())
    n_workers = len(workers)

    if model_type == "classification":
        X, y = make_classification(
            n_samples=n_workers * 10000,
            n_features=20,
            n_informative=15,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=999,
        )
        y = y.astype(np.int32)
    else:
        X, y = make_regression(
            n_samples=n_workers * 10000,
            n_features=20,
            n_informative=5,
            random_state=123,
        )
        y = y.astype(np.float32)
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 100, random_state=123
    )

    X_train_df, y_train_df = _prep_training_data(client, X_train, y_train, 1)
    X_test_dask_array = from_array(X_test)

    n_estimators = n_workers * 8

    if model_type == "classification":
        cuml_mod = cuRFC_mg(
            n_estimators=n_estimators,
            max_depth=8,
            n_bins=16,
            ignore_empty_partitions=True,
        )
        cuml_mod.fit(X_train_df, y_train_df, broadcast_data=fit_broadcast)
        cuml_mod_predict = cuml_mod.predict(
            X_test_dask_array, broadcast_data=transform_broadcast
        )

        cuml_mod_predict = cuml_mod_predict.compute()
        cuml_mod_predict = cp.asnumpy(cuml_mod_predict)
        acc_score = accuracy_score(cuml_mod_predict, y_test, normalize=True)
        assert acc_score >= 0.68

    else:
        cuml_mod = cuRFR_mg(
            n_estimators=n_estimators,
            max_depth=8,
            n_bins=16,
            ignore_empty_partitions=True,
        )
        cuml_mod.fit(X_train_df, y_train_df, broadcast_data=fit_broadcast)
        cuml_mod_predict = cuml_mod.predict(
            X_test_dask_array, broadcast_data=transform_broadcast
        )

        cuml_mod_predict = cuml_mod_predict.compute()
        cuml_mod_predict = cp.asnumpy(cuml_mod_predict)
        acc_score = r2_score(y_test, cuml_mod_predict)
        assert acc_score >= 0.72

    if transform_broadcast:
        assert cuml_mod.internal_model is None
