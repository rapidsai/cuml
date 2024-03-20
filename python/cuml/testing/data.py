#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
from typing import TYPE_CHECKING, TypeVar

from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    _T = TypeVar("_T", np.ndarray, cp.ndarray)
else:
    np = cpu_only_import("numpy")
    cp = gpu_only_import("cupy")
    _T = TypeVar("_T")


def rosenbrock(x: _T, a: float = 1, b: float = 100) -> _T:
    """Famous Rosenbrock example"""

    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def g_rosenbrock(x: _T, a: float = 1, b: float = 100) -> _T:
    """Gradietn of rosenbrock example"""

    if isinstance(x, np.ndarray):
        array = np.array
    else:
        array = cp.array
    g = array(
        [
            -2 * a - 4 * b * x[0] * (-x[0] ** 2 + x[1]) + 2 * x[0],
            b * (-2 * x[0] ** 2 + 2 * x[1]),
        ]
    )

    return g
