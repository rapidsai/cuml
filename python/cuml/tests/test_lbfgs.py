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
from typing import TYPE_CHECKING, Tuple

from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
from cuml.solvers.lbfgs import _fmin_lbfgs
from cuml.testing.data import g_rosenbrock, rosenbrock

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
else:
    np = cpu_only_import("numpy")
    cp = gpu_only_import("cupy")


def test_lbfgs() -> None:
    rng = np.random.default_rng(42)

    a = cp.asarray(rng.normal(1, scale=0.1, size=1))
    b = cp.asarray(rng.normal(100, scale=10, size=1))

    def f(x: np.ndarray) -> Tuple[float, np.ndarray]:
        fb = rosenbrock(x, a, b)
        g = g_rosenbrock(x, a, b)
        return fb, g

    x0 = cp.zeros(2)
    res = _fmin_lbfgs(f, x0)

    # analytical minimum
    res_true = cp.array([a[0], a[0] ** 2])

    cp.testing.assert_allclose(res.x, res_true, rtol=1e-4)
