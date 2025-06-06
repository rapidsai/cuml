#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import cupy
import pytest
from sklearn.datasets import load_iris

import cuml
from cuml.datasets import make_classification, make_regression
from cuml.model_selection import GridSearchCV, train_test_split
from cuml.pipeline import Pipeline, make_pipeline
from cuml.preprocessing import StandardScaler
from cuml.svm import SVC
from cuml.testing.utils import ClassEnumerator


def test_pipeline():
    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("svc", SVC())])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    assert score > 0.75


def test_gridsearchCV():
    iris = load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    clf = GridSearchCV(SVC(), parameters)
    clf.fit(iris.data, iris.target)
    assert clf.best_params_["kernel"] == "rbf"
    assert clf.best_params_["C"] == 10


@pytest.fixture(scope="session")
def regression_dataset(request):
    X, y = make_regression(n_samples=10, n_features=5, random_state=0)
    return train_test_split(X, y, random_state=0)


@pytest.fixture(scope="session")
def classification_dataset(request):
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    return train_test_split(X, y, random_state=0)


models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()


@pytest.mark.parametrize(
    "model_key",
    [
        "ElasticNet",
        "Lasso",
        "Ridge",
        "LinearRegression",
        "LogisticRegression",
        "MBSGDRegressor",
        "RandomForestRegressor",
        "KNeighborsRegressor",
    ],
)
@pytest.mark.parametrize("instantiation", ["Pipeline", "make_pipeline"])
def test_pipeline_with_regression(
    regression_dataset, model_key, instantiation
):
    X_train, X_test, y_train, y_test = regression_dataset
    model_const = models[model_key]
    if model_key == "RandomForestRegressor":
        model = model_const(n_bins=2)
    else:
        model = model_const()

    if instantiation == "Pipeline":
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
    elif instantiation == "make_pipeline":
        pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X_train, y_train)
    prediction = pipe.predict(X_test)
    assert isinstance(prediction, cupy.ndarray)
    _ = pipe.score(X_test, y_test)


@pytest.mark.parametrize(
    "model_key",
    ["MBSGDClassifier", "RandomForestClassifier", "KNeighborsClassifier"],
)
@pytest.mark.parametrize("instantiation", ["Pipeline", "make_pipeline"])
def test_pipeline_with_classification(
    classification_dataset, model_key, instantiation
):
    X_train, X_test, y_train, y_test = classification_dataset
    model_const = models[model_key]
    if model_key == "RandomForestClassifier":
        model = model_const(n_bins=2)
    else:
        model = model_const()
    if instantiation == "Pipeline":
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
    elif instantiation == "make_pipeline":
        pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X_train, y_train)
    prediction = pipe.predict(X_test)
    assert isinstance(prediction, cupy.ndarray)
    if model_key == "RandomForestClassifier":
        pytest.skip(
            "RandomForestClassifier is not yet supported "
            "by the Pipeline utility"
        )
    _ = pipe.score(X_test, y_test)
