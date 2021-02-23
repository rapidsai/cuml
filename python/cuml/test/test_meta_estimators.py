#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cuml.pipeline import Pipeline
from cuml.model_selection import GridSearchCV

from cuml.datasets import make_classification
from cuml.model_selection import train_test_split
from sklearn.datasets import load_iris

from cuml.experimental.preprocessing import StandardScaler
from cuml.svm import SVC


def test_pipeline():
    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    pipe.fit(X_train, y_train)
    Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
    score = pipe.score(X_test, y_test)
    assert score > 0.8


def test_gridsearchCV():
    iris = load_iris()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    clf = GridSearchCV(SVC(), parameters)
    clf.fit(iris.data, iris.target)
    assert clf.best_params_['kernel'] == 'rbf'
    assert clf.best_params_['C'] == 10
