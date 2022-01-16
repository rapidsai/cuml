# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cupy import linalg
import numpy as np
import cudf
from cuml import KernelRidge as cuKernelRidge
from cuml.test.utils import (
    array_equal,
    small_regression_dataset,
    small_classification_dataset,
    unit_param,
    quality_param,
    stress_param,
)
from scipy.sparse.construct import rand
from sklearn.metrics import pairwise
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge as skKernelRidge
from hypothesis import note,given, settings, strategies as st


def gradient_norm(X, y, model):
    X = cp.array(X)
    y = cp.array(y)
    K = cp.array(model._get_kernel(X.get()))
    w = cp.array(model.dual_coef_).reshape(y.shape)
    grad = -cp.dot(K, y)
    grad += cp.dot(cp.dot(K, K), w)
    grad += cp.dot(K * model.alpha, w)
    return linalg.norm(grad)


standard_kernels = sorted(pairwise.PAIRWISE_KERNEL_FUNCTIONS.keys())
@given(st.sampled_from(standard_kernels))
@settings(deadline=5000)
def test_kernel_ridge(kernel):
    model = cuKernelRidge(kernel=kernel,gamma=1.0)
    skl_model = skKernelRidge(kernel=kernel,gamma=1.0)
    X, y = make_regression(20, random_state=2)

    if kernel == "chi2" or kernel == "additive_chi2":
        # X must be positive
        X = X + abs(X.min()) +1.0
    
    model.fit(X, y)
    skl_model.fit(X, y)
    # For a convex optimisation problem we should arrive at gradient norm 0
    # If the solution has converged correctly
    assert gradient_norm(X, y, skl_model) < 1e-5
    assert gradient_norm(X, y, model) < 1e-5
