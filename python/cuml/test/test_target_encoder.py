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

from cuml.preprocessing.TargetEncoder import TargetEncoder
import cudf
import numpy as np
from cuml.test.utils import array_equal


def test_targetencoder_fit_transform():
    train = cudf.DataFrame({'category': ['a', 'b', 'b', 'a'],
                            'label': [1, 0, 1, 1]})
    encoder = TargetEncoder()
    train_encoded = encoder.fit_transform(train.category, train.label)
    answer = np.array([1., 1., 0., 1.])
    assert array_equal(train_encoded, answer)

    encoder = TargetEncoder()
    encoder.fit(train.category, train.label)
    train_encoded = encoder.transform(train.category)

    assert array_equal(train_encoded, answer)


def test_targetencoder_transform():
    """
    Note that there are newly-encountered values in test,
    namely, 'c' and 'd'.
    """
    train = cudf.DataFrame({'category': ['a', 'b', 'b', 'a'],
                            'label': [1, 0, 1, 1]})
    test = cudf.DataFrame({'category': ['c', 'b', 'a', 'd']})
    encoder = TargetEncoder()
    encoder.fit_transform(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    answer = np.array([0.75, 0.5, 1., 0.75])
    assert array_equal(test_encoded, answer)

    encoder = TargetEncoder()
    encoder.fit(train.category, train.label)
    test_encoded = encoder.transform(test.category)
    assert array_equal(test_encoded, answer)


def test_one_category():
    train = cudf.DataFrame({'category': ['a', 'a', 'a', 'a'],
                            'label': [3, 0, 0, 3]})
    test = cudf.DataFrame({'category': ['c', 'b', 'a', 'd']})

    encoder = TargetEncoder()
    train_encoded = encoder.fit_transform(train.category, train.label)
    answer = np.array([1., 2., 2., 1.])
    assert array_equal(train_encoded, answer)

    test_encoded = encoder.transform(test.category)
    answer = np.array([1.5, 1.5, 1.5, 1.5])
    assert array_equal(test_encoded, answer)

    assert array_equal(test_encoded, answer)
