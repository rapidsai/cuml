
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

from cuml.dask.ensemble import RandomForestClassifier as cuRFC_mg
from cuml.dask.ensemble import RandomForestRegressor as cuRFR_mg
from cuml.dask.common import utils as dask_utils
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from dask.distributed import Client

import dask_cudf
import cudf
import numpy as np
import pandas as pd


@pytest.mark.parametrize('partitions_per_worker', [1, 3])
def test_rf_classification_dask(partitions_per_worker, cluster):

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    c = Client(cluster)

    try:

        X, y = make_classification(n_samples=10000, n_features=20,
                                   n_clusters_per_class=1, n_informative=10,
                                   random_state=123, n_classes=5)

        X = X.astype(np.float32)
        y = y.astype(np.int32)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=1000)

        cu_rf_params = {
            'n_estimators': 25,
            'max_depth': 13,
            'n_bins': 15,
        }

        workers = c.has_what().keys()
        n_partitions = partitions_per_worker * len(workers)
        X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
        X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

        y_cudf = np.array(pd.DataFrame(y_train).values)
        y_cudf = y_cudf[:, 0]
        y_cudf = cudf.Series(y_cudf)
        y_train_df = \
            dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

        X_train_df, \
            train_df = dask_utils.persist_across_workers(c,
                                                         [X_train_df,
                                                          y_train_df],
                                                         workers=workers)
        cu_rf_mg = cuRFC_mg(**cu_rf_params)
        cu_rf_mg.fit(X_train_df, y_train_df)
        cu_rf_mg_predict = cu_rf_mg.predict(X_test)

        acc_score = accuracy_score(cu_rf_mg_predict, y_test, normalize=True)

        assert acc_score > 0.8

    finally:
        c.close()


@pytest.mark.parametrize('partitions_per_worker', [1, 3])
def test_rf_regression_dask(partitions_per_worker, cluster):

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    c = Client(cluster)

    try:

        X, y = make_regression(n_samples=40000, n_features=20,
                               n_informative=10, random_state=123)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=1000)

        cu_rf_params = {
            'n_estimators': 25,
            'max_depth': 13,
        }

        workers = c.has_what().keys()
        n_partitions = partitions_per_worker * len(workers)

        X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
        X_train_df = \
            dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

        y_cudf = np.array(pd.DataFrame(y_train).values)
        y_cudf = y_cudf[:, 0]
        y_cudf = cudf.Series(y_cudf)
        y_train_df = \
            dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

        X_train_df, y_train_df = dask_utils.persist_across_workers(
            c, [X_train_df, y_train_df], workers=workers)

        cu_rf_mg = cuRFR_mg(**cu_rf_params)
        cu_rf_mg.fit(X_train_df, y_train_df)
        cu_rf_mg_predict = cu_rf_mg.predict(X_test)

        acc_score = r2_score(cu_rf_mg_predict, y_test)

        print(str(acc_score))

        assert acc_score >= 0.70

    finally:
        c.close()
