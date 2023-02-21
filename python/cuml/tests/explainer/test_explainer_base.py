#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml import LinearRegression as cuLR
from cuml.explainer.base import SHAPBase
from pylibraft.common.handle import Handle
import pytest
from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


@pytest.mark.parametrize("handle", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
@pytest.mark.parametrize("order", ["C", None])
def test_init_explainer_base_init_cuml_model(handle, dtype, order):
    bg = np.arange(10).reshape(5, 2).astype(np.float32)
    y = np.arange(5).astype(np.float32)
    bg_df = cudf.DataFrame(bg)

    model = cuLR().fit(bg, y)

    if handle:
        handle = Handle()
    else:
        handle = None

    explainer = SHAPBase(
        model=model.predict,
        background=bg_df,
        order=order,
        link="identity",
        verbose=2,
        random_state=None,
        is_gpu_model=None,
        handle=handle,
        dtype=None,
        output_type=None,
    )

    assert explainer.ncols == 2
    assert explainer.nrows == 5
    assert np.all(cp.asnumpy(explainer.background) == bg)
    assert np.all(explainer.feature_names == bg_df.columns)
    assert explainer.is_gpu_model

    # check that we infer the order from the model (F for LinearRegression) if
    # it is not passed explicitly
    if order is None:
        assert explainer.order == "F"
    else:
        assert explainer.order == order

    # check that we keep the model's handle if one is not passed explicitly
    if handle is not None:
        assert explainer.handle == handle
    else:
        assert explainer.handle == model.handle


@pytest.mark.parametrize("handle", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
@pytest.mark.parametrize("order", ["C", None])
@pytest.mark.parametrize("is_gpu_model", [True, False, None])
@pytest.mark.parametrize("output_type", ["cupy", None])
def test_init_explainer_base_init_abritrary_model(
    handle, dtype, order, is_gpu_model, output_type
):
    bg = np.arange(10).reshape(5, 2).astype(np.float32)

    if handle:
        handle = Handle()
    else:
        handle = None

    explainer = SHAPBase(
        model=dummy_func,
        background=bg,
        order=order,
        order_default="F",
        link="identity",
        verbose=2,
        random_state=None,
        is_gpu_model=is_gpu_model,
        handle=handle,
        dtype=None,
        output_type=output_type,
    )

    assert explainer.ncols == 2
    assert explainer.nrows == 5
    assert np.all(cp.asnumpy(explainer.background) == bg)
    if not is_gpu_model or is_gpu_model is None:
        assert not explainer.is_gpu_model
    else:
        assert explainer.is_gpu_model

    if output_type is not None:
        assert explainer.output_type == output_type
    else:
        assert explainer.output_type == "numpy"

    # check that explainer defaults to order_default is order is not passed
    # explicitly
    if order is None:
        assert explainer.order == "F"
    else:
        assert explainer.order == order

    # check that we keep the model's handle if one is not passed explicitly
    if handle is not None:
        assert explainer.handle == handle
    else:
        isinstance(explainer.handle, Handle)


def test_init_explainer_base_wrong_dtype():

    with pytest.raises(ValueError):
        explainer = SHAPBase(
            model=dummy_func, background=np.ones(10), dtype=np.int32
        )
        explainer.ncols


def dummy_func(x):
    return x
