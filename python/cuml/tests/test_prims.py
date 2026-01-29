# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest

from cuml.prims.label import make_monotonic
from cuml.testing.utils import array_equal


@pytest.mark.parametrize("arr_type", ["np", "cp"])
@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("copy", [True, False])
def test_monotonic_without_classes(arr_type, dtype, copy):
    """Test make_monotonic when classes are derived from labels."""
    arr = np.array([0, 15, 10, 50, 20, 50], dtype=dtype)

    if arr_type == "cp":
        arr = cp.asarray(arr, dtype=dtype)

    arr_orig = cp.asarray(arr).copy()

    monotonic, returned_classes = make_monotonic(arr, copy=copy)

    cp.cuda.Stream.null.synchronize()

    # Verify monotonic mapping: [0, 15, 10, 50, 20, 50] -> [0, 2, 1, 4, 3, 4]
    # (sorted unique: 0->0, 10->1, 15->2, 20->3, 50->4)
    expected_monotonic = cp.array([0, 2, 1, 4, 3, 4], dtype=dtype)
    assert array_equal(monotonic, expected_monotonic)

    # Verify returned classes are sorted unique values
    expected_classes = cp.array([0, 10, 15, 20, 50], dtype=dtype)
    assert array_equal(returned_classes, expected_classes)

    # Verify dtype is preserved
    assert monotonic.dtype == dtype

    # Check in-place behavior (only meaningful for device arrays)
    if arr_type == "cp":
        if copy:
            assert array_equal(arr, arr_orig)
        else:
            assert array_equal(arr, monotonic)


@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
def test_monotonic_inversion(dtype):
    """Test that monotonic labels can be inverted back to original values."""
    original = cp.array([0, 15, 10, 50, 20, 50], dtype=dtype)

    monotonic, classes = make_monotonic(original, copy=True)

    # Invert: use classes array to map indices back to original values
    inverted = classes[monotonic]

    cp.cuda.Stream.null.synchronize()

    assert array_equal(inverted, original)


@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("copy", [True, False])
def test_monotonic_with_explicit_classes(dtype, copy):
    """Test make_monotonic when explicit classes are provided."""
    # Classes in non-sorted order
    classes = cp.array([5, 2, 8], dtype=dtype)
    labels = cp.array([8, 2, 5, 2, 8], dtype=dtype)
    labels_orig = labels.copy()

    monotonic, returned_classes = make_monotonic(
        labels, classes=classes, copy=copy
    )

    cp.cuda.Stream.null.synchronize()

    # Labels should map to their position in the original classes array
    # 5 -> 0, 2 -> 1, 8 -> 2
    expected = cp.array([2, 1, 0, 1, 2], dtype=dtype)
    assert array_equal(monotonic, expected)
    assert array_equal(returned_classes, classes)

    # Verify dtype is preserved
    assert monotonic.dtype == dtype

    # Check in-place behavior
    if copy:
        assert array_equal(labels, labels_orig)
    else:
        assert array_equal(labels, monotonic)


@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
def test_monotonic_unknown_labels(dtype):
    """Test that labels not in classes map to len(classes)."""
    classes = cp.array([1, 2, 3], dtype=dtype)
    labels = cp.array([1, 999, 2, 3, -1], dtype=dtype)

    monotonic, _ = make_monotonic(labels, classes=classes, copy=True)

    cp.cuda.Stream.null.synchronize()

    # Unknown labels (999, -1) should map to len(classes) = 3
    # 1 -> 0, 999 -> 3, 2 -> 1, 3 -> 2, -1 -> 3
    expected = cp.array([0, 3, 1, 2, 3], dtype=dtype)
    assert array_equal(monotonic, expected)
    assert monotonic.dtype == dtype


def test_monotonic_dtype_preservation():
    """Test that output dtype matches input labels dtype."""
    # Test int32 labels with int64 classes
    labels_32 = cp.array([1, 2, 3], dtype=cp.int32)
    classes_64 = cp.array([1, 2, 3], dtype=cp.int64)

    monotonic, _ = make_monotonic(labels_32, classes=classes_64, copy=True)
    assert monotonic.dtype == cp.int32

    # Test int64 labels with int32 classes
    labels_64 = cp.array([1, 2, 3], dtype=cp.int64)
    classes_32 = cp.array([1, 2, 3], dtype=cp.int32)

    monotonic, _ = make_monotonic(labels_64, classes=classes_32, copy=True)
    assert monotonic.dtype == cp.int64

    # Test without explicit classes
    labels_32 = cp.array([5, 10, 15], dtype=cp.int32)
    monotonic, _ = make_monotonic(labels_32, copy=True)
    assert monotonic.dtype == cp.int32
