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
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, List, Tuple, Optional

from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
else:
    np = cpu_only_import("numpy")
    cp = gpu_only_import("cupy")


def _vlbfgs(
    grad: np.ndarray, history: List[np.ndarray], y: List[np.ndarray]
) -> np.ndarray:
    """VL-BFGS from [1].

    References
    ----------
    .. [1] Weizhu Chen, Zhenghao Wang, Jingren Zhou.
          "Large-scale L-BFGS using MapReduce".
          Advances in Neural Information Processing Systems Vol, 27 (2014)
    """
    n = len(grad)
    m = len(history)
    if m == 0:
        return -grad

    b = cp.array(history + y + [grad])
    assert b.shape == (2 * m + 1, n), (b.shape, m)
    # pre-compute the dot product.
    B = b @ b.T
    assert B.shape == (2 * m + 1, 2 * m + 1)

    B[(B >= 0.0) & (B < 1e-15)] = 1e-6
    B[(B < 0.0) & (B > -1e-15)] = -1e-6

    delta = cp.zeros(len(b))
    delta[-1] = -1
    alpha = cp.zeros(len(b))

    for i in reversed(range(m)):
        # linear combination of b
        alpha[i] = delta @ B[:, i] / B[i, m + i]
        delta[m + i] = delta[m + i] - alpha[i]

    delta = delta * B[m - 1, 2 * m - 1] / B[2 * m - 1, 2 * m - 1]

    for i in range(m):
        beta = delta.dot(B[:, i + m]) / B[i, i + m]
        delta[i] = delta[i] + (alpha[i] - beta)

    return delta @ b


def _line_search(
    f: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    loss: cp.ndarray,
    grad: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """backtracking line search."""
    alpha = 1.0
    c = 1e-4
    rho = 0.5

    new_loss, new_g = f(x + alpha * d)
    beta = c * (grad @ d)
    # iterate until the sufficient decrease condition is met
    while new_loss > (loss + alpha * beta) and alpha * rho > 1e-15:
        alpha *= rho
        new_loss, new_g = f(x + alpha * d)
        new_g = cp.asarray(new_g.flatten())
    return alpha, new_g


@dataclass
class _LbgfsResult:
    x: np.ndarray  # optimized parameters
    loss: float  # final loss value from the objective
    norm: float  # gradient norm
    n_iters: int  # number of iterations used


def _fmin_lbfgs(
    f: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    gtol: float = 1e-5,
    max_iter: Optional[int] = None,
) -> _LbgfsResult:
    """An internal implementation of L-BGFS, not intended for the public use.

    Parameters
    ----------

    f :
        The objective function to minimize. It should return a scalar value array
        containing the function value, and an array for gradient.

    x0 :
        Initial point for the parameters.

    gtol :
        Condition for stopping the optimization.

    max_iter :
        Maximum number of iterations.

    """

    @wraps(f)
    def in_cupy(x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        loss, g = f(x)
        return cp.asarray(loss), cp.array(g.flatten())

    fn = in_cupy

    if max_iter is None:
        # configuration from scipy.optimize
        max_iter = len(x0) * 200

    x = x0
    loss, g = fn(x)
    if g.ndim != 1 and g.shape[1] != 1:
        raise ValueError("gradient should be a vector.")
    g = cp.asarray(g.flatten())

    history: List[np.ndarray] = []
    y: List[np.ndarray] = []

    for i in range(max_iter):
        # direction
        p = _vlbfgs(g, history, y)
        step, new_g = _line_search(fn, loss, g, x, p)

        norm = cp.linalg.norm(new_g)

        delta = p * step
        x = x + delta
        history.append(delta)
        y.append(new_g - g)

        g = new_g
        if step < 1e-6:
            history = []
            y = []

        if norm < gtol:
            break

        if i > len(history):
            history.pop(0)
            y.pop(0)

    return _LbgfsResult(x, loss, norm, i + 1)
