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

import numpy as np
import pytest

from cuml.linear_model import MBSGDClassifier as cumlMBSGClassifier
from cuml.test.utils import unit_param, quality_param, stress_param

from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module", params=[
    unit_param([500, 20, 10, np.float32]),
    unit_param([500, 20, 10, np.float64]),
    quality_param([5000, 100, 50, np.float32]),
    quality_param([5000, 100, 50, np.float64]),
    stress_param([500000, 1000, 500, np.float32]),
    stress_param([500000, 1000, 500, np.float64]),
], ids=['500-20-10-f32', '500-20-10-f64',
        '5000-100-50-f32', '5000-100-50-f64',
        '500000-1000-500-f32', '500000-1000-500-f64'])
def make_dataset(request):
    nrows, ncols, n_info, datatype = request.param
    X, y = make_classification(n_samples=nrows, n_informative=n_info,
                               n_features=ncols, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=10)

    y_train = y_train.astype(datatype)
    y_test = y_test.astype(datatype)

    return nrows, X_train, X_test, y_train, y_test


@pytest.mark.parametrize(
    # Grouped those tests to reduce the total number of individual tests
    # while still keeping good coverage of the different features of MBSGD
    ('lrate', 'penalty', 'loss'), [
        ('constant', 'none', 'log'),
        ('invscaling', 'l2', 'hinge'),
        ('adaptive', 'l1', 'squared_loss'),
        ('constant', 'elasticnet', 'hinge'),
    ]
)
def test_mbsgd_classifier(lrate, penalty, loss, make_dataset):
    nrows, X_train, X_test, y_train, y_test = make_dataset

    cu_mbsgd_classifier = cumlMBSGClassifier(learning_rate=lrate, eta0=0.005,
                                             epochs=100, fit_intercept=True,
                                             batch_size=2, tol=0.0,
                                             penalty=penalty)

    cu_mbsgd_classifier.fit(X_train, y_train)
    cu_pred = cu_mbsgd_classifier.predict(X_test).to_array()
    cu_acc = accuracy_score(cu_pred, y_test)

    if nrows < 500000:
        skl_sgd_classifier = SGDClassifier(learning_rate=lrate, eta0=0.005,
                                           max_iter=100, fit_intercept=True,
                                           tol=0.0, penalty=penalty,
                                           random_state=0)

        skl_sgd_classifier.fit(X_train, y_train)
        skl_pred = skl_sgd_classifier.predict(X_test)
        skl_acc = accuracy_score(skl_pred, y_test)
        assert cu_acc >= skl_acc - 0.06


def test_mbsgd_classifier_default(make_dataset):
    nrows, X_train, X_test, y_train, y_test = make_dataset

    cu_mbsgd_classifier = cumlMBSGClassifier()

    cu_mbsgd_classifier.fit(X_train, y_train)
    cu_pred = cu_mbsgd_classifier.predict(X_test).to_array()
    cu_acc = accuracy_score(cu_pred, y_test)

    if nrows < 500000:
        skl_sgd_classifier = SGDClassifier()

        skl_sgd_classifier.fit(X_train, y_train)
        skl_pred = skl_sgd_classifier.predict(X_test)
        skl_acc = accuracy_score(skl_pred, y_test)
        assert cu_acc >= skl_acc - 0.05
