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

from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
from typing import Union

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")

ArrayType = Union[np.ndarray, cp.ndarray]


def array_equal(
    actual: ArrayType,
    expected: ArrayType,
    rtol: float = 0.0,
    atol: float = 0.0,
    strict_type: bool = False,
    err_msg: str = "Arrays are not equal",
    verbose: bool = True,
) -> None:
    """
    Compare two arrays with optional tolerance and type checks.

    Parameters
    ----------
    actual : np.ndarray or cp.ndarray
        The actual array to test.
    expected : np.ndarray or cp.ndarray
        The expected array to compare against.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    strict_type : bool, optional
        If True, raises if the array types differ (e.g., np vs cp).
    err_msg : str, optional
        Error message to display on failure.
    verbose : bool, optional
        Whether to include difference summary on failure.

    Raises
    ------
    AssertionError
        If arrays are not equal within the given tolerance or types mismatch.
    """
    if strict_type and type(actual) is not type(expected):
        raise AssertionError(
            f"{err_msg}: Type mismatch {type(actual)} vs {type(expected)}"
        )

    if cp and (
        isinstance(actual, cp.ndarray) or isinstance(expected, cp.ndarray)
    ):
        actual = cp.asnumpy(actual)
        expected = cp.asnumpy(expected)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
        verbose=verbose,
    )
