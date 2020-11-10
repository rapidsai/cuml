#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
from sklearn.feature_extraction.text import CountVectorizer


def pytest_configure(config):
    cp.cuda.set_allocator(None)


@pytest.fixture(scope="module")
def nlp_20news():
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True,
                                      random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


def pytest_addoption(parser):
    parser.addoption("--run_stress",
                     action="store_true",
                     default=False,
                     help="run stress tests")

    parser.addoption("--run_quality",
                     action="store_true",
                     default=False,
                     help="run quality tests")

    parser.addoption("--run_unit",
                     action="store_true",
                     default=False,
                     help="run unit tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_quality"):
        # --run_quality given in cli: do not skip quality tests
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

        return

    else:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if config.getoption("--run_stress"):
        # --run_stress given in cli: do not skip stress tests

        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

        return

    else:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
