# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest

from cuml.prims.label import check_labels, invert_labels, make_monotonic
from cuml.testing.utils import array_equal


@pytest.mark.parametrize("arr_type", ["np", "cp"])
@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("copy", [True, False])
def test_monotonic_validate_invert_labels(arr_type, dtype, copy):
    arr = np.array([0, 15, 10, 50, 20, 50], dtype=dtype)

    original = arr.copy()

    if arr_type == "cp":
        arr = cp.asarray(arr, dtype=dtype)
        arr_orig = arr.copy()

    monotonic, mapped_classes = make_monotonic(arr, copy=copy)

    cp.cuda.Stream.null.synchronize()

    assert array_equal(monotonic, np.array([0, 2, 1, 4, 3, 4]))

    # We only care about in-place updating if data is on device
    if arr_type == "cp":
        if copy:
            assert array_equal(arr_orig, arr)
        else:
            assert array_equal(arr, monotonic)

    wrong_classes = cp.asarray([0, 1, 2], dtype=dtype)
    val_labels = check_labels(monotonic, classes=wrong_classes)

    cp.cuda.Stream.null.synchronize()

    assert not val_labels

    correct_classes = cp.asarray([0, 1, 2, 3, 4], dtype=dtype)
    val_labels = check_labels(monotonic, classes=correct_classes)

    cp.cuda.Stream.null.synchronize()

    assert val_labels

    if arr_type == "cp":
        monotonic_copy = monotonic.copy()

    inverted = invert_labels(
        monotonic,
        classes=cp.asarray([0, 10, 15, 20, 50], dtype=dtype),
        copy=copy,
    )

    cp.cuda.Stream.null.synchronize()

    if arr_type == "cp":
        if copy:
            assert array_equal(monotonic_copy, monotonic)
        else:
            assert array_equal(monotonic, arr_orig)

    assert array_equal(inverted, original)


@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
def test_labels_with_many_classes(dtype):
    """Test label functions work when classes exceed shared memory limit."""
    n_classes = 15000  # Exceeds ~48KB shared memory limit
    labels = cp.random.randint(0, n_classes, size=35000, dtype=dtype)
    classes = cp.arange(n_classes, dtype=dtype)

    assert check_labels(labels, classes)
    assert not check_labels(labels, cp.arange(100, dtype=dtype))

    monotonic, mapped = make_monotonic(labels.copy())
    inverted = invert_labels(monotonic, mapped)
    assert array_equal(inverted, labels)
