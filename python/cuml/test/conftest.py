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

    import rmm

    max_pool_size = int(4.95 * (1 << 30))

    # Align down to 256 (required by RMM)
    max_pool_size = max_pool_size & ~(256 - 1)

    # TEMP: Set the max memory pool that is used to 4.95 GiB for 1/7 A100
    rmm.reinitialize(pool_allocator=True,
                     managed_memory=True,
                     initial_pool_size=max_pool_size // 2,
                     maximum_pool_size=max_pool_size)


@pytest.fixture(scope="module")
def nlp_20news():
    try:
        twenty_train = fetch_20newsgroups(subset='train',
                                          shuffle=True,
                                          random_state=42)
    except IOError:
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y
