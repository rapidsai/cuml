#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
import numpy as np
import pytest
from pylibraft.common.handle import Handle
from sklearn.linear_model import LinearRegression as skreg

import cuml
from cuml import PCA
from cuml import LinearRegression as reg
from cuml.datasets import make_regression
from cuml.explainer.common import (
    get_cai_ptr,
    get_handle_from_cuml_model_func,
    get_link_fn_from_str_or_fn,
    get_tag_from_model_func,
    link_dict,
    model_func_call,
)
from cuml.testing.utils import ClassEnumerator

models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()

_default_tags = [
    "preferred_input_order",
    "X_types_gpu",
    "non_deterministic",
    "requires_positive_X",
    "requires_positive_y",
    "X_types",
    "poor_score",
    "no_validation",
    "multioutput",
    "allow_nan",
    "stateless",
    "multilabel",
    "_skip_test",
    "_xfail_checks",
    "multioutput_only",
    "binary_only",
    "requires_fit",
    "requires_y",
    "pairwise",
]


def test_get_gpu_tag_from_model_func():
    # test getting the gpu tags from the model that we use in explainers
    model = reg()

    order = get_tag_from_model_func(
        func=model.predict, tag="preferred_input_order", default="C"
    )

    assert order == "F"

    out_types = get_tag_from_model_func(
        func=model.predict, tag="X_types_gpu", default=False
    )

    assert isinstance(out_types, list)
    assert "2darray" in out_types

    # checking arbitrary function
    order = get_tag_from_model_func(
        func=dummy_func, tag="preferred_input_order", default="C"
    )

    assert order == "C"

    out_types = get_tag_from_model_func(
        func=dummy_func, tag="X_types_gpu", default=False
    )

    assert out_types is False

    model2 = skreg()

    out_types = get_tag_from_model_func(
        func=model2.predict, tag="X_types_gpu", default=False
    )

    assert out_types is False


@pytest.mark.parametrize("model", list(models.values()))
def test_get_tag_from_model_func(model):
    mod = create_dummy_model(model)

    for tag in _default_tags:
        res = get_tag_from_model_func(
            func=mod._get_param_names, tag=tag, default="FFF"
        )

        if tag != "preferred_input_order":
            assert res != "FFF"


@pytest.mark.parametrize("model", list(models.values()))
def test_get_handle_from_cuml_model_func(model):
    mod = create_dummy_model(model)

    handle = get_handle_from_cuml_model_func(
        mod._get_param_names, create_new=True
    )

    assert isinstance(handle, Handle)


@pytest.mark.parametrize("create_new", [True, False])
def test_get_handle_from_dummy_func(create_new):
    handle = get_handle_from_cuml_model_func(dummy_func, create_new=create_new)

    res = isinstance(handle, Handle)

    assert res == create_new


def test_model_func_call_gpu():
    X, y = make_regression(
        n_samples=81,
        n_features=10,
        noise=0.1,
        random_state=42,
        dtype=np.float32,
    )

    model = reg().fit(X, y)

    z = model_func_call(X=X, model_func=model.predict, gpu_model=True)

    assert isinstance(z, cp.ndarray)

    z = model_func_call(
        X=cp.asnumpy(X), model_func=dummy_func, gpu_model=False
    )

    assert isinstance(z, cp.ndarray)

    with pytest.raises(TypeError):
        z = model_func_call(X=X, model_func=dummy_func, gpu_model=True)

    model = PCA(n_components=10).fit(X)

    z = model_func_call(X=X, model_func=model.transform, gpu_model=True)

    assert isinstance(z, cp.ndarray)


def test_get_cai_ptr():
    a = cp.ones(10)
    ptr = get_cai_ptr(a)

    assert ptr == a.__cuda_array_interface__["data"][0]

    b = np.ones(10)
    with pytest.raises(TypeError):
        ptr = get_cai_ptr(b)


@pytest.mark.parametrize("link_function", ["identity", "logit"])
def test_get_link_fn_from_str(link_function):
    fn = get_link_fn_from_str_or_fn(link_function)
    a = cp.ones(10)

    assert cp.all(fn(a) == link_dict[link_function](a))
    assert cp.all(fn.inverse(a) == link_dict[link_function].inverse(a))


def test_get_link_fn_from_wrong_str():
    with pytest.raises(ValueError):
        get_link_fn_from_str_or_fn("this_is_wrong")


def test_get_link_fn_from_fn():
    def dummylink(x):
        return 2 * x

    # check we raise error if link has no inverse
    with pytest.raises(TypeError):
        get_link_fn_from_str_or_fn(dummylink)

    def dummylink_inv(x):
        return x / 2

    dummylink.inverse = dummylink_inv

    fn = get_link_fn_from_str_or_fn(dummylink)

    assert fn(2) == 4
    assert fn.inverse(2) == 1


def create_dummy_model(model):
    try:
        mod = model()
    except TypeError:
        mod = model(np.zeros(10))
    return mod


def dummy_func(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array")
    return np.mean(x, axis=1)
