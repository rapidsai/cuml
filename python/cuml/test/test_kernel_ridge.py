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
from numba import cuda
from cuml import (
    KernelRidge as cuKernelRidge,
    pairwise_kernels,
    PAIRWISE_KERNEL_FUNCTIONS,
)
import pytest
import math
import inspect
from cuml.test.utils import (
    array_equal,
    small_regression_dataset,
    small_classification_dataset,
    unit_param,
    quality_param,
    stress_param,
)
from scipy.sparse.construct import rand
from sklearn.metrics import pairwise, mean_squared_error as mse
from sklearn.datasets import make_regression
from hypothesis import note, given, settings, strategies as st
from hypothesis.extra.numpy import arrays


def gradient_norm(X, y, model):
    X = cp.array(X)
    y = cp.array(y)
    K = cp.array(model._get_kernel(X.get()))
    w = cp.array(model.dual_coef_).reshape(y.shape)
    grad = -cp.dot(K, y)
    grad += cp.dot(cp.dot(K, K), w)
    grad += cp.dot(K * model.alpha, w)
    return linalg.norm(grad)


def test_pairwise_kernels_basic():
    X = np.zeros((4, 4))
    # standard kernel with no argument
    pairwise_kernels(X, metric="chi2")
    pairwise_kernels(X, metric="linear")
    # standard kernel with correct kwd argument
    pairwise_kernels(X, metric="chi2", gamma=1.0)
    # standard kernel with incorrect kwd argument
    with pytest.raises(
        ValueError, match="kwds contains arguments not used by kernel function"
    ):
        pairwise_kernels(X, metric="chi2", wrong_parameter_name=1.0)
    # standard kernel with filtered kwd argument
    pairwise_kernels(X, metric="chi2", filter_params=True, wrong_parameter_name=1.0)

    # incorrect function type
    def non_numba_kernel(x, y):
        return x.dot(y)

    with pytest.raises(
        TypeError, match="Kernel function should be a numba device function."
    ):
        pairwise_kernels(X, metric=non_numba_kernel)

    # correct function type
    @cuda.jit(device=True)
    def numba_kernel(x, y, special_argument=3.0):
        return 1 + 2

    pairwise_kernels(X, metric=numba_kernel)
    pairwise_kernels(X, metric=numba_kernel, special_argument=1.0)

    # malformed function
    @cuda.jit(device=True)
    def bad_numba_kernel(x):
        return 1 + 2

    with pytest.raises(
        ValueError, match="Expected at least two arguments to kernel function."
    ):
        pairwise_kernels(X, metric=bad_numba_kernel)

    # malformed function 2 - No default value
    @cuda.jit(device=True)
    def bad_numba_kernel2(x, y, z):
        return 1 + 2

    with pytest.raises(
        ValueError, match="Extra kernel parameters must be passed as keyword arguments."
    ):
        pairwise_kernels(X, metric=bad_numba_kernel2)

    # Precomputed
    assert array_equal(X, pairwise_kernels(X, metric="precomputed"))


@cuda.jit(device=True)
def custom_kernel(x, y, custom_arg=5.0):
    sum = 0.0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    return math.exp(-custom_arg * sum) + 0.1


test_kernels = sorted(pairwise.PAIRWISE_KERNEL_FUNCTIONS.keys()) + [custom_kernel]


@st.composite
def kernel_arg_strategy(draw):
    kernel = draw(st.sampled_from(test_kernels))
    kernel_func = (
        PAIRWISE_KERNEL_FUNCTIONS[kernel] if isinstance(kernel, str) else kernel
    )
    # Inspect the function and generate some arguments
    all_func_kwargs = list(inspect.signature(kernel_func.py_func).parameters.values())[
        2:
    ]
    param = {}
    for arg in all_func_kwargs:
        # 50% chance we generate this parameter or leave it as default
        if draw(st.booleans()):
            continue
        if isinstance(arg.default, float) or arg.default is None:
            param[arg.name] = draw(st.floats(0.0, 5.0))
        if isinstance(arg.default, int):
            param[arg.name] = draw(st.integers(0, 5))

    return (kernel, param)


@st.composite
def array_strategy(draw):
    X_m = draw(st.integers(1, 20))
    X_n = draw(st.integers(1, 10))
    dtype = draw(st.sampled_from([np.float64
            , np.float32]))
    X = draw(
        arrays(
            dtype=dtype,
            shape=(X_m, X_n),
            elements=st.floats(0, 5, width=32),
        )
    )
    if draw(st.booleans()):
        Y_m = draw(st.integers(1, 20))
        Y = draw(
            arrays(
            dtype=dtype,
                shape=(Y_m, X_n),
                elements=st.floats(0, 5, width=32),
            )
        )
    else:
        Y = None
    return (X, Y)


@given(kernel_arg_strategy(), array_strategy())
@settings(deadline=5000, max_examples=20)
def test_pairwise_kernels(kernel_arg, XY):
    X, Y = XY
    kernel, args = kernel_arg
    K = pairwise_kernels(X, Y, metric=kernel, **args)
    skl_kernel = kernel.py_func if hasattr(kernel, "py_func") else kernel
    K_sklearn = pairwise.pairwise_kernels(X, Y, metric=skl_kernel,**args)
    assert np.allclose(K, K_sklearn, rtol=0.01)


@given(kernel_arg_strategy())
@settings(deadline=5000, max_examples=100)
def test_estimator(kernel_arg):
    kernel, args = kernel_arg
    model = cuKernelRidge(kernel=kernel, kernel_params=args)
    X, y = make_regression(20, random_state=2)

    if kernel == "chi2" or kernel == "additive_chi2":
        # X must be positive
        X = X + abs(X.min()) + 1.0

    model.fit(X, y)
    # For a convex optimisation problem we should arrive at gradient norm 0
    # If the solution has converged correctly
    assert gradient_norm(X, y, model) < 1e-5

    # model should do better than predicting average
    pred = model.predict(X).get()
    baseline_pred = np.full_like(pred, y.mean())
    assert mse(y,pred) < mse(y, baseline_pred)
