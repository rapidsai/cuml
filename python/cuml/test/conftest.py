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
import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_california_housing
from sklearn.feature_extraction.text import CountVectorizer


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
