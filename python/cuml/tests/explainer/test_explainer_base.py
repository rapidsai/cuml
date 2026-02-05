#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numpy as np
import pytest

from cuml import LinearRegression as cuLR
from cuml.explainer.base import SHAPBase


@pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
@pytest.mark.parametrize("order", ["C", None])
def test_init_explainer_base_init_cuml_model(dtype, order):
    bg = np.arange(10).reshape(5, 2).astype(np.float32)
    y = np.arange(5).astype(np.float32)
    bg_df = cudf.DataFrame(bg)

    model = cuLR().fit(bg, y)

    explainer = SHAPBase(
        model=model.predict,
        background=bg_df,
        order=order,
        link="identity",
        verbose=2,
        random_state=None,
        is_gpu_model=None,
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
@pytest.mark.parametrize("order", ["C", None])
@pytest.mark.parametrize("is_gpu_model", [True, False, None])
@pytest.mark.parametrize("output_type", ["cupy", None])
def test_init_explainer_base_init_abritrary_model(
    dtype, order, is_gpu_model, output_type
):
    bg = np.arange(10).reshape(5, 2).astype(np.float32)

    explainer = SHAPBase(
        model=dummy_func,
        background=bg,
        order=order,
        order_default="F",
        link="identity",
        verbose=2,
        random_state=None,
        is_gpu_model=is_gpu_model,
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


def test_init_explainer_base_wrong_dtype():
    with pytest.raises(ValueError):
        explainer = SHAPBase(
            model=dummy_func, background=np.ones(10), dtype=np.int32
        )
        explainer.ncols


def dummy_func(x):
    return x
