# Copyright (c) 2020, NVIDIA CORPORATION.
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
from cuml.common import input_to_cuml_array
from sklearn.datasets import make_classification
import numpy as np


np_X_cl, np_y_cl = make_classification(n_samples=500,
                                       n_features=20,
                                       n_clusters_per_class=1,
                                       n_informative=12,
                                       random_state=123, n_classes=5)

np_X_int = np.random.randint(100, size=(500, 20)).astype(np.float64)
np_X_int.ravel()[np.random.choice(np_X_int.size,
                                  int(np_X_int.size*0.02),
                                  replace=False)] = np.nan


@pytest.fixture(scope="session",
                params=["numpy", "dataframe", "cupy", "cudf", "numba"])
def small_clf_dataset(request):
    elms = tuple(map(lambda x: input_to_cuml_array(x)[0], [np_X_cl, np_y_cl]))
    X, y = tuple(map(lambda x: x.to_output(request.param), elms))
    return (np_X_cl, np_y_cl), (X, y)


@pytest.fixture(scope="session",
                params=["numpy", "dataframe", "cupy", "cudf", "numba"])
def small_int_dataset(request):
    X = input_to_cuml_array(np_X_int)[0]
    X = X.to_output(request.param)
    return np_X_int, X


def assert_array_equal(x, y, mean_diff_tol=0.0, max_diff_tol=None,
                       ratio_diff_tol=None):
    if x.shape != y.shape:
        raise ValueError('Shape mismatch')

    n_elements = x.size

    diff = np.abs(x - y)
    mean_diff = np.nanmean(diff)
    max_diff = np.nanmax(diff)
    ratio_diff = np.nansum(diff != 0) / n_elements

    if (mean_diff_tol is not None and mean_diff > mean_diff_tol) or \
       (max_diff_tol is not None and max_diff > max_diff_tol) or \
       (ratio_diff_tol is not None and ratio_diff > ratio_diff_tol):
        err_msg = """Too much difference:\n\t
                     Mean diff: {}\n\t
                     Max diff: {}\n\t
                     Ratio of diff: {}"""
        raise ValueError(err_msg.format(mean_diff, max_diff, ratio_diff))
