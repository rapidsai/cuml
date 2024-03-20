#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
import cupy as cp
import pytest
from sklearn.model_selection import cross_val_predict as skl_cvp

from cuml.datasets import make_classification, make_regression
from cuml.internals.import_utils import has_xgboost
from cuml.linear_model import LinearRegression
from cuml.model_selection import StratifiedKFold, cross_val_predict


def test_cross_val_predict_regression() -> None:
    X, y = make_regression(512, 16, random_state=3)
    estimator = LinearRegression()

    predictions = cross_val_predict(estimator, X, y, cv=3)
    p1 = skl_cvp(estimator, X, y, cv=3)
    cp.testing.assert_allclose(predictions, p1)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_with_xgboost() -> None:
    import xgboost as xgb
    n_samples = 128
    n_estimators = 10

    for device in ("cuda", "cpu"):
        clf = xgb.XGBClassifier(device=device, n_estimators=n_estimators)

        X, y = make_classification(n_samples=n_samples, random_state=3)
        p0 = cross_val_predict(clf, X, y, cv=5, method="predict_proba")
        # make sure it's using stratified folds
        kfold = StratifiedKFold(n_splits=5)

        indices = []
        predictions = []
        for train, test in kfold.split(X.get(), y.get()):
            clf.fit(X[train], y[train])
            indices.append(cp.asarray(test))
            y_predt = clf.predict_proba(X[test])
            predictions.append(cp.asarray(y_predt))

        indices_array = cp.concatenate(indices)
        inv = cp.empty(len(indices_array), dtype=cp.int64)
        inv[indices_array]= cp.arange(len(indices_array))

        predt_array = cp.concatenate(predictions, axis=0)
        predt_array = predt_array[inv]
        cp.testing.assert_allclose(predt_array, p0)

        reg = xgb.XGBRegressor(device=device, n_estimators=n_estimators)
        X, y = make_regression(n_samples=n_samples, random_state=3)
        p0 = cross_val_predict(reg, X, y, cv=5)
        p1 = skl_cvp(reg, cp.asnumpy(X), cp.asnumpy(y), cv=5)
        cp.testing.assert_allclose(p0, p1)
