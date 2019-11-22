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

import pytest
from cuml.preprocessing import LabelBinarizer
from cuml.test.utils import array_equal

import numpy as np
import cupy as cp


@pytest.mark.parametrize(
    "labels", [([1, 4, 5, 2, 0, 1, 6, 2, 3, 4],
                [4, 2, 6, 3, 2, 0, 1]),
               ([9, 8, 2, 1, 3, 4],
                [8, 2, 1, 2, 2])]
)
def test_basic_functions(labels):

    # @todo: Test sparse output, test different inputs

    fit_labels, xform_labels = labels

    binarizer = LabelBinarizer()
    binarizer.fit(fit_labels)

    assert array_equal(cp.asnumpy(binarizer.classes_),
                       np.unique(cp.asnumpy(fit_labels)))

    xformed = binarizer.transform(xform_labels)

    assert xformed.shape[1] == binarizer.classes_.shape[0]

    original = binarizer.inverse_transform(xformed)

    assert array_equal(cp.asnumpy(original),
                       xform_labels)
