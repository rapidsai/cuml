# Copyright (c) 2025, NVIDIA CORPORATION.
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

import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=2,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="module")
def multiclass():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


def test_svc(binary):
    X, y = binary
    svc = SVC().fit(X, y)
    assert svc.score(X, y) > 0.5


def test_svc_probability(binary):
    X, y = binary
    svc = SVC(probability=True).fit(X, y)
    # Inference and score works
    assert svc.score(X, y) > 0.5

    # probA_ and probB_ exist and are the correct shape
    assert svc.probA_.shape == (1,)
    assert svc.probB_.shape == (1,)

    # predict_proba works
    y_pred = svc.predict_proba(X).argmax(axis=1)
    assert accuracy_score(y_pred, y) > 0.5


def test_svc_multiclass(multiclass):
    X, y = multiclass
    svr = SVC().fit(X, y)
    assert svr.score(X, y) > 0.5


def test_conditional_methods():
    svc = SVC()
    assert not hasattr(svc, "predict_proba")
    assert not hasattr(svc, "predict_log_proba")

    svc.probability = True
    assert hasattr(svc, "predict_proba")
    assert hasattr(svc, "predict_log_proba")
    # Ensure these methods aren't forwarded CPU attributes
    assert svc.predict_proba is not svc._cpu.predict_proba
    assert svc.predict_log_proba is not svc._cpu.predict_log_proba
