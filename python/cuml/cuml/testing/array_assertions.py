# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
from typing import Union

import numpy as np

from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")

ArrayType = Union[np.ndarray, cp.ndarray]


def array_equal(a, b, tol=1e-6, relative_diff=False, report_summary=False):

    if hasattr(a, "get"):
        a = a.get()
    if hasattr(b, "get"):
        b = b.get()

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise AssertionError(f"Shapes differ: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)

    if relative_diff:
        idx = np.nonzero(np.abs(b) > tol)
        diff[idx] = diff[idx] / np.abs(b[idx])

    if not np.all(diff <= tol):
        if report_summary:
            idx = np.argsort(diff.ravel())
            print("Largest diffs:")
            a_flat = a.ravel()
            b_flat = b.ravel()
            diff_flat = diff.ravel()
            for i in idx[-5:]:
                if diff_flat[i] > tol:
                    print(
                        diff_flat[i], "at", i, "values", a_flat[i], b_flat[i]
                    )
            print(
                "Avgdiff:",
                np.mean(diff),
                "stddiff:",
                np.std(diff),
                "avgval:",
                np.mean(b),
            )
        raise AssertionError("Arrays are not equal within tolerance.")

    return True
