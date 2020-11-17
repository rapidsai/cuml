#
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

import numpy as np
import pytest

from cuml import LinearRegression as reg
from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_tag_from_model_func
from sklearn.datasets import make_regression
# todo: uncomment after PR 3113 is merged
# from cuml.common.base import _default_tags


_default_tags = [
    'preferred_input_order',
    'X_types_gpu',
    'non_deterministic',
    'requires_positive_X',
    'requires_positive_y',
    'X_types',
    'poor_score',
    'no_validation',
    'multioutput',
    'allow_nan',
    'stateless',
    'multilabel',
    '_skip_test',
    '_xfail_checks',
    'multioutput_only',
    'binary_only',
    'requires_fit',
    'requires_y',
    'pairwise'
]


def test_get_dtype_from_model_func():
    X, y = make_regression(n_samples=81, n_features=10, noise=0.1,
                           random_state=42)

    # checking model with float32 dtype
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model_f32 = reg().fit(X, y)

    assert get_dtype_from_model_func(model_f32.predict) == np.float32

    # checking model with float64 dtype
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    model_f64 = reg().fit(X, y)

    assert get_dtype_from_model_func(model_f64.predict) == np.float64

    # checking model that has not been fitted yet
    model_not_fit = reg()

    assert(get_dtype_from_model_func(model_not_fit.predict) is None)

    # checking arbitrary function
    def dummy_func(x):
        return x + x

    assert get_dtype_from_model_func(dummy_func) is None


def test_get_gpu_tag_from_model_func():
    pytest.skip("Skipped until tags PR "
                "https://github.com/rapidsai/cuml/pull/3113 is merged")

    # testing getting the gpu tags from the model that we use in explainers

    model = reg()

    order = get_tag_from_model_func(func=model.predict,
                                    tag='preferred_input_order',
                                    default='C')

    assert order == 'F'

    out_types = get_tag_from_model_func(func=model.predict,
                                        tag='X_types_gpu',
                                        default=False)

    assert isinstance(out_types, list)
    assert '2darray' in out_types

    # checking arbitrary function
    def dummy_func(x):
        return x + x

    order = get_tag_from_model_func(func=dummy_func,
                                    tag='preferred_input_order',
                                    default='C')

    assert order == 'C'

    out_types = get_tag_from_model_func(func=dummy_func,
                                        tag='X_types_gpu',
                                        default=False)

    assert out_types is False


@pytest.mark.parametrize("tag", list(_default_tags))
def test_get_tag_from_model_func(tag):
    pytest.skip("Skipped until tags PR "
                "https://github.com/rapidsai/cuml/pull/3113 is merged")

    model = reg()

    res = get_tag_from_model_func(func=model.predict,
                                  tag='preferred_input_order',
                                  default='FFF')

    assert res != 'FFF'
