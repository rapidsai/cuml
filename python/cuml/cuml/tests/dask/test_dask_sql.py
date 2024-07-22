# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from cuml.internals.import_utils import has_dask_sql
from cuml.internals.safe_imports import cpu_only_import
import pytest
import cuml
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
np = cpu_only_import("numpy")


if has_dask_sql():
    from dask_sql import Context
else:
    pytest.skip("Dask-SQL not available", allow_module_level=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("nrows", [100000])
@pytest.mark.parametrize("ncols", [20, 50])
@pytest.mark.parametrize("n_parts", [2, 20])
@pytest.mark.parametrize("wrap_predict", [True, False])
def test_dask_sql_sg_logistic_regression(
    datatype, nrows, ncols, n_parts, wrap_predict
):
    if wrap_predict:
        cuml.set_global_output_type("input")
    else:
        cuml.set_global_output_type("cudf")

    X, y = make_classification(
        n_samples=nrows, n_features=ncols, n_informative=5, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_df = cudf.DataFrame(
        X_train, dtype=datatype, columns=[chr(i) for i in range(ncols)]
    )
    train_df["target"] = y_train
    train_ddf = dask_cudf.from_cudf(train_df, npartitions=n_parts)

    c = Context()
    c.create_table("train_df", train_ddf)

    train_query = f"""
        CREATE MODEL model WITH (
            model_class = 'cuml.linear_model.LogisticRegression',
            wrap_predict = {wrap_predict},
            target_column = 'target'
        ) AS (
            SELECT * FROM train_df
        )
    """

    c.sql(train_query)

    skmodel = LogisticRegression().fit(X_train, y_train)

    test_df = cudf.DataFrame(
        X_test, dtype=datatype, columns=[chr(i) for i in range(ncols)]
    )
    test_ddf = dask_cudf.from_cudf(test_df, npartitions=n_parts)
    c.create_table("test_df", test_ddf)

    inference_query = """
        SELECT * FROM PREDICT(
            MODEL model,
            SELECT * FROM test_df
        )
    """

    preds = c.sql(inference_query).compute()
    score = cuml.metrics.accuracy_score(y_test, preds["target"].to_numpy())

    assert score >= skmodel.score(X_test, y_test) - 0.022
