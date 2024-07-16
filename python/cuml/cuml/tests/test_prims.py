# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import cpu_only_import
from cuml.prims.label import make_monotonic
from cuml.prims.label import invert_labels
from cuml.prims.label import check_labels

from cuml.testing.utils import array_equal

import pytest

from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


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
