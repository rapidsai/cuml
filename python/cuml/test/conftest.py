#
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
import numpy as np
import pytest

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_classification as skl_make_clas
from sklearn.datasets import make_regression as skl_make_reg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def pytest_configure(config):
    cp.cuda.set_allocator(None)


@pytest.fixture(scope="module")
def nlp_20news():
    try:
        twenty_train = fetch_20newsgroups(subset='train',
                                          shuffle=True,
                                          random_state=42)
    except:  # noqa E722
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


@pytest.fixture(scope="module")
def housing_dataset():
    try:
        data = fetch_california_housing()

    # failing to download has appeared as multiple varied errors in CI
    except:  # noqa E722
        pytest.xfail(reason="Error fetching housing dataset")

    X = cp.array(data['data'])
    y = cp.array(data['target'])

    feature_names = data['feature_names']

    return X, y, feature_names


def create_synthetic_dataset(generator=skl_make_reg,
                             n_samples=100,
                             n_features=10,
                             test_size=0.25,
                             random_state_generator=None,
                             random_state_train_test_split=None,
                             dtype=np.float32,
                             **kwargs):
    X, y = generator(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state_generator,
        **kwargs
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state_train_test_split
    )

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def exact_shap_regression_dataset():
    return create_synthetic_dataset(generator=skl_make_reg,
                                    n_samples=101,
                                    n_features=11,
                                    test_size=3,
                                    random_state_generator=42,
                                    random_state_train_test_split=42,
                                    noise=0.1)


@pytest.fixture(scope="module")
def exact_shap_classification_dataset():
    return create_synthetic_dataset(generator=skl_make_clas,
                                    n_samples=101,
                                    n_features=11,
                                    test_size=3,
                                    random_state_generator=42,
                                    random_state_train_test_split=42)
