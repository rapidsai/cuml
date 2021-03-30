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

import pytest

import numpy as np
from cuml._thirdparty.sklearn.utils.validation import check_X_y


def test_check_X_y():  # noqa: F811
    X = np.ones((100, 10))
    y1 = np.ones((100,))
    y2 = np.ones((100, 1))
    y3 = np.ones((100, 2))
    y4 = np.ones((101,))

    check_X_y(X, y1, multi_output=False)
    check_X_y(X, y2, multi_output=False)
    with pytest.raises(Exception):
        check_X_y(X, y3, multi_output=False)
    with pytest.raises(Exception):
        check_X_y(X, y4, multi_output=False)
