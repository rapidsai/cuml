#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

from cuml.common import has_scipy
import cuml.internals.logger as logger
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import_from,
    null_decorator,
)

nvtx_annotate = gpu_only_import_from("nvtx", "annotate", alt=null_decorator)
np = cpu_only_import("numpy")


def _fd_fprime(x, f, h):
    """(internal) Computes finite difference."""
    g = np.zeros(len(x))
    for i in range(len(x)):
        xph = np.copy(x)
        xmh = np.copy(x)
        xph[i] += h
        xmh[i] -= h
        fph = f(xph)
        fmh = f(xmh)
        g[i] = (fph - fmh) / (2 * h)

    return g


@nvtx_annotate(message="LBFGS", domain="cuml_python")
def batched_fmin_lbfgs_b(
    func,
    x0,
    num_batches,
    fprime=None,
    args=(),
    bounds=None,
    m=10,
    factr=1e7,
    pgtol=1e-5,
    epsilon=1e-8,
    iprint=-1,
    maxiter=15000,
    maxls=20,
):
    """A batch-aware L-BFGS-B implementation to minimize a loss function `f` given
    an initial set of parameters `x0`.

    Parameters
    ----------
    func : function (x: array) -> array[M] (M = n_batches)
           The function to minimize. The function should return an array of
           size = `num_batches`
    x0 : array
         Starting parameters
    fprime : function (x: array) -> array[M*n_params] (optional)
             The gradient. Should return an array of derivatives for each
             parameter over batches.
             When omitted, uses Finite-differencing to estimate the gradient.
    args   : Tuple
             Additional arguments to func and fprime
    bounds : List[Tuple[float, float]]
             Box-constrains on the parameters
    m      : int
             L-BFGS parameter: number of previous arrays to store when
             estimating inverse Hessian.
    factr  : float
             Stopping criterion when function evaluation not progressing.
             Stop when `|f(xk+1) - f(xk)| < factor*eps_mach`
             where `eps_mach` is the machine precision
    pgtol  : float
             Stopping criterion when gradient is sufficiently "flat".
             Stop when |grad| < pgtol.
    epsilon : float
              Finite differencing step size when approximating `fprime`
    iprint : int
             -1 for no diagnostic info
             n=1-100 for diagnostic info every n steps.
             >100 for detailed diagnostic info
             Only used for Scipy < 1.15
    maxiter : int
              Maximum number of L-BFGS iterations
    maxls   : int
              Maximum number of line-search iterations.

    """

    if has_scipy():
        from scipy.optimize import _lbfgsb

        scipy_greater_115 = has_scipy(min_version="1.15")
    else:
        raise RuntimeError("Scipy is needed to run batched_fmin_lbfgs_b")

    n = len(x0) // num_batches

    if fprime is None:

        def fprime_f(x):
            return _fd_fprime(x, func, epsilon)

        fprime = fprime_f

    if bounds is None:
        bounds = [(None, None)] * n

    nbd = np.zeros(n, np.int32)
    low_bnd = np.zeros(n, np.float64)
    upper_bnd = np.zeros(n, np.float64)
    bounds_map = {(None, None): 0, (1, None): 1, (1, 1): 2, (None, 1): 3}
    for i in range(0, n):
        lb, ub = bounds[i]
        if lb is not None:
            low_bnd[i] = lb
            lb = 1
        if ub is not None:
            upper_bnd[i] = ub
            ub = 1
        nbd[i] = bounds_map[lb, ub]

    # working arrays needed by L-BFGS-B implementation in SciPy.
    # One for each series
    x = [
        np.copy(np.array(x0[ib * n : (ib + 1) * n], np.float64))
        for ib in range(num_batches)
    ]
    f = [np.copy(np.array(0.0, np.float64)) for ib in range(num_batches)]
    g = [np.copy(np.zeros((n,), np.float64)) for ib in range(num_batches)]
    wa = [
        np.copy(np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, np.float64))
        for ib in range(num_batches)
    ]
    iwa = [np.copy(np.zeros(3 * n, np.int32)) for ib in range(num_batches)]

    # we need different inputs after Scipy 1.15 using a C-based lbfgs
    if scipy_greater_115:
        task = [np.copy(np.zeros(1, np.int32)) for ib in range(num_batches)]
        ln_task = [np.copy(np.zeros(1, np.int32)) for ib in range(num_batches)]
    else:
        task = [np.copy(np.zeros(1, "S60")) for ib in range(num_batches)]
        csave = [np.copy(np.zeros(1, "S60")) for ib in range(num_batches)]

    lsave = [np.copy(np.zeros(4, np.int32)) for ib in range(num_batches)]
    isave = [np.copy(np.zeros(44, np.int32)) for ib in range(num_batches)]
    dsave = [np.copy(np.zeros(29, np.float64)) for ib in range(num_batches)]
    if not scipy_greater_115:
        for ib in range(num_batches):
            task[ib][:] = "START"

    n_iterations = np.zeros(num_batches, dtype=np.int32)

    converged = num_batches * [False]

    warn_flag = np.zeros(num_batches)

    while not all(converged):
        with nvtx_annotate("LBFGS-ITERATION", domain="cuml_python"):
            for ib in range(num_batches):
                if converged[ib]:
                    continue
                if scipy_greater_115:
                    _lbfgsb.setulb(
                        m,
                        x[ib],
                        low_bnd,
                        upper_bnd,
                        nbd,
                        f[ib],
                        g[ib],
                        factr,
                        pgtol,
                        wa[ib],
                        iwa[ib],
                        task[ib],
                        lsave[ib],
                        isave[ib],
                        dsave[ib],
                        maxls,
                        ln_task[ib],
                    )
                else:
                    _lbfgsb.setulb(
                        m,
                        x[ib],
                        low_bnd,
                        upper_bnd,
                        nbd,
                        f[ib],
                        g[ib],
                        factr,
                        pgtol,
                        wa[ib],
                        iwa[ib],
                        task[ib],
                        iprint,
                        csave[ib],
                        lsave[ib],
                        isave[ib],
                        dsave[ib],
                        maxls,
                    )

            xk = np.concatenate(x)
            fk = func(xk)
            gk = fprime(xk)
            for ib in range(num_batches):
                if converged[ib]:
                    continue

                # This are the status messages in scipy 1.15:
                # status_messages = {
                #     0 : "START",
                #     1 : "NEW_X",
                #     2 : "RESTART",
                #     3 : "FG",
                #     4 : "CONVERGENCE",
                #     5 : "STOP",
                #     6 : "WARNING",
                #     7 : "ERROR",
                #     8 : "ABNORMAL"
                # }
                if scipy_greater_115:
                    cond1 = task[0] == 3
                    cond2 = task[0] == 1
                    cond3 = task[0] == 4
                else:
                    task_str = task[ib].tobytes()
                    task_str_strip = task[ib].tobytes().strip(b"\x00").strip()
                    cond1 = task_str.startswith(b"FG")
                    cond2 = task_str.startswith(b"NEW_X")
                    cond3 = task_str_strip.startswith(b"CONV")

                if cond1:
                    # needs function evaluation
                    f[ib] = fk[ib]
                    g[ib] = gk[ib * n : (ib + 1) * n]
                elif cond2:
                    n_iterations[ib] += 1
                    if n_iterations[ib] >= maxiter:
                        if scipy_greater_115:
                            task[ib][0] = 5
                            task[ib][1] = 504
                        else:
                            task[ib][
                                :
                            ] = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
                elif cond3:
                    converged[ib] = True
                    warn_flag[ib] = 0
                else:
                    converged[ib] = True
                    warn_flag[ib] = 2
                    continue

    xk = np.concatenate(x)

    if iprint > 0:
        logger.info(
            "CONVERGED in ({}-{}) iterations (|\\/f|={})".format(
                np.min(n_iterations),
                np.max(n_iterations),
                np.linalg.norm(fprime(xk), np.inf),
            )
        )

        if (warn_flag > 0).any():
            for ib in range(num_batches):
                if warn_flag[ib] > 0:
                    logger.info(
                        "WARNING: id={} convergence issue: {}".format(
                            ib, task[ib].tobytes()
                        )
                    )

    return xk, n_iterations, warn_flag
