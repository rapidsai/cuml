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
from sklearn.calibration import CalibratedClassifierCV as sklcccv

from cuml.calibration import CalibratedClassifierCV, _estimate_sigmoid
from cuml.datasets import make_classification
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC


def valid_multi_classes(est, X):
    assert isinstance(est.classes_, cp.ndarray)
    cp.testing.assert_allclose(est.classes_, cp.asarray([0, 1, 2, 3, 4]))

    proba = est.predict_proba(X)
    s = proba.sum(axis=1)
    cp.testing.assert_allclose(s, 1.0)

    y_pred = est.predict(X)
    cp.testing.assert_allclose(proba.argmax(axis=1), y_pred)


def valid_binary(est, X):
    assert isinstance(est.classes_, cp.ndarray)
    cp.testing.assert_allclose(est.classes_, cp.asarray([0, 1]))

    proba = est.predict_proba(X)
    s = proba.sum(axis=1)
    cp.testing.assert_allclose(s, 1.0)

    y_pred = est.predict(X)
    cp.testing.assert_allclose(proba.argmax(axis=1), y_pred)



def test_sigmoid() -> None:
    X, y = make_classification(random_state=3)
    model = _estimate_sigmoid(y, y, None)
    assert model.slope < -7.5 and model.intercept > 3.5


def test_calibration_prefit() -> None:
    # We use logistic regression and SVC for testing estimator with the
    # `decision_function` method, random forest for testing estimators with the
    # `predict_proba` method. The implementation prioritizes `decision_function` over
    # `predict_proba`.

    assert hasattr(SVC(), "decision_function")
    assert hasattr(LogisticRegression(), "decision_function")
    assert not hasattr(RandomForestClassifier(), "decision_function")

    # - binary, no weight, decision
    X, y = make_classification(random_state=3)

    clf = LogisticRegression()
    clf.fit(X, y)

    est = sklcccv(clf, cv="prefit")  # compare with sklearn
    est.fit(X.get(), y.get())
    proba_0 = est.predict_proba(X.get())

    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y)
    assert isinstance(est.classes_, cp.ndarray)
    cp.testing.assert_allclose(est.classes_, cp.asarray([0, 1]))

    proba_1 = est.predict_proba(X)
    assert proba_1.shape[1] == 2
    s = proba_1[:, 0] + proba_1[:, 1]
    cp.testing.assert_allclose(s, 1.0)
    cp.testing.assert_allclose(proba_0, proba_1, rtol=1e-2)

    # - binary, weight, decision
    rng = cp.random.default_rng(3)
    sample_weight = rng.uniform(0.0, 1.0, size=X.shape[0])
    clf = LogisticRegression()
    clf.fit(X, y)
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y, sample_weight=sample_weight)
    assert isinstance(est.classes_, cp.ndarray)
    cp.testing.assert_allclose(est.classes_, cp.asarray([0, 1]))
    proba_2 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_2.sum(axis=1), 1.0)
    assert not cp.allclose(proba_1, proba_2, rtol=1e-3)

    sample_weight = cp.full_like(sample_weight, 1.0)
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y, sample_weight=sample_weight)
    proba_2 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_2, proba_1, rtol=1e-5)

    # - binary, no weight, predict_proba
    clf = RandomForestClassifier()
    clf.fit(X, y)
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y)
    proba_1 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_1.sum(axis=1), 1.0)

    # - binary, weight, predict_proba
    sample_weight = rng.uniform(0.0, 1.0, size=X.shape[0])
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y, sample_weight=sample_weight)
    proba_2 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_2.sum(axis=1), 1.0)
    assert not cp.allclose(proba_1, proba_2, rtol=1e-3)
    sample_weight = cp.full_like(sample_weight, 1.0)
    est.fit(X, y, sample_weight=sample_weight)
    proba_2 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_2, proba_1, rtol=1e-5)

    # - multi class, no weight, predict_proba
    X, y = make_classification(random_state=3, n_classes=5, n_features=8, n_informative=5)
    clf = RandomForestClassifier()
    clf.fit(X, y)

    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y)
    valid_multi_classes(est, X)
    proba_nw = est.predict_proba(X)

    est = sklcccv(clf, cv="prefit")  # compare with sklearn
    est.fit(X.get(), y.get())
    proba_skl = est.predict_proba(X.get())
    cp.testing.assert_allclose(cp.asarray(proba_skl), proba_nw, rtol=1e-3)

    # - multi class, weigth, predict_proba
    sample_weight = rng.uniform(0.0, 1.0, size=X.shape[0])
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y, sample_weight=sample_weight)

    valid_multi_classes(est, X)
    proba_2 = est.predict_proba(X)
    assert not cp.allclose(proba_nw, proba_2, rtol=1e-3)

    sample_weight = cp.full_like(sample_weight, 1.0)
    est.fit(X, y, sample_weight=sample_weight)
    proba_2 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_2, proba_nw, rtol=1e-5)

    # - multi class, no weight, decision
    # Change from logistic to SVC: https://github.com/rapidsai/cuml/issues/5741
    clf = SVC()
    clf.fit(X, y)

    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y)
    valid_multi_classes(est, X)
    proba_nw = est.predict_proba(X)

    # - multi class weight, decision
    sample_weight = rng.uniform(0.0, 1.0, size=X.shape[0])
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y, sample_weight=sample_weight)
    valid_multi_classes(est, X)

    proba_2 = est.predict_proba(X)
    assert not cp.allclose(proba_nw, proba_2, rtol=1e-3)

    sample_weight = cp.full_like(sample_weight, 1.0)
    est = CalibratedClassifierCV(estimator=clf, cv="prefit")
    est.fit(X, y, sample_weight=sample_weight)
    proba_2 = est.predict_proba(X)
    cp.testing.assert_allclose(proba_2, proba_nw, rtol=1e-5)


def valid_sample_weight(est: CalibratedClassifierCV, X, y, proba_nw) -> CalibratedClassifierCV:
    rng = cp.random.default_rng(3)
    sample_weight = rng.uniform(0.0, 1.0, size=X.shape[0])
    est.fit(X, y, sample_weight=sample_weight)
    proba = est.predict_proba(X)
    assert not cp.allclose(proba_nw, proba, rtol=1e-3)

    sample_weight = cp.full_like(sample_weight, 1.0)
    est.fit(X, y, sample_weight=sample_weight)
    proba = est.predict_proba(X)
    cp.testing.assert_allclose(proba, proba_nw, rtol=1e-5)

    return est


@pytest.mark.parametrize("ensemble", [True, False])
def test_calibration_cv(ensemble: bool) -> None:
    # - binary, no weight, decision
    X, y = make_classification(random_state=3)
    clf = LogisticRegression()
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    est.fit(X, y)
    proba = est.predict_proba(X)

    assert isinstance(est.classes_, cp.ndarray)
    cp.testing.assert_allclose(est.classes_, cp.array([0, 1]))
    cp.testing.assert_allclose(proba.sum(axis=1), 1.0)

    est = sklcccv(estimator=clf, cv=3, ensemble=ensemble)  # compare with skl
    est.fit(X.get(), y.get())
    proba_skl = est.predict_proba(X.get())
    if ensemble is True:
        cp.testing.assert_allclose(
            proba.argmax(axis=1), cp.asarray(proba_skl).argmax(axis=1)
        )

    # - binary, weight, decision
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    valid_sample_weight(est, X, y, proba)

    # - binary, no weight, predict_proba
    clf = RandomForestClassifier()
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    est.fit(X, y)
    proba = est.predict_proba(X)
    cp.testing.assert_allclose(proba.sum(axis=1), 1.0)

    # - binary, weight, predict_proba
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    valid_sample_weight(est, X, y, proba)

    # -- multi class

    X, y = make_classification(random_state=3, n_classes=5, n_features=8, n_informative=5)
    clf = RandomForestClassifier()
    # - multi class, no weight, predict_proba
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    est.fit(X, y)
    valid_multi_classes(est, X)
    proba = est.predict_proba(X)
    cp.testing.assert_allclose(proba.sum(axis=1), 1.0)

    # - multi class, weigth, predict_proba
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    est.fit(X, y)
    est = valid_sample_weight(est, X, y, proba)
    valid_multi_classes(est, X)
    proba = est.predict_proba(X)
    cp.testing.assert_allclose(proba.sum(axis=1), 1.0)

    clf = SVC()
    # - multi class, no weight, decision
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    est.fit(X, y)
    valid_multi_classes(est, X)
    proba = est.predict_proba(X)
    cp.testing.assert_allclose(proba.sum(axis=1), 1.0)

    # - multi class weight, decision
    est = CalibratedClassifierCV(estimator=clf, cv=3, ensemble=ensemble)
    est.fit(X, y)
    valid_multi_classes(est, X)
    est = valid_sample_weight(est, X, y, proba)
    proba = est.predict_proba(X)
    cp.testing.assert_allclose(proba.sum(axis=1), 1.0)
