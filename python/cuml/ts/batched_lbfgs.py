import scipy.optimize as optimize
import numpy as np
from IPython.core.debugger import set_trace
from scipy.optimize import _lbfgsb
from .batched_kalman import pynvtx_range_push, pynvtx_range_pop

from collections import deque

def _fd_fprime(x, f, h):
    g = np.zeros(len(x))
    for i in range(len(x)):
        xph = np.copy(x)
        xmh = np.copy(x)
        xph[i] += h
        xmh[i] -= h
        fph = f(xph)
        fmh = f(xmh)
        g[i] = (fph - fmh)/(2*h)

    return g

def batched_fmin_lbfgs_b(func, x0, num_batches, fprime=None, args=(),
                         approx_grad=0,
                         bounds=None, m=10, factr=1e7, pgtol=1e-5,
                         epsilon=1e-8,
                         iprint=-1, maxfun=15000, maxiter=15000, disp=None,
                         callback=None, maxls=20):

    pynvtx_range_push("LBFGS")
    n = len(x0) // num_batches

    if fprime is None and approx_grad is True:
        fprime = lambda x: _fd_fprime(x, func, epsilon)

    if bounds is None:
        bounds = [(None, None)] * n

    nbd = np.zeros(n, np.int32)
    low_bnd = np.zeros(n, np.float64)
    upper_bnd = np.zeros(n, np.float64)
    bounds_map = {(None, None): 0,
                  (1, None): 1,
                  (1, 1): 2,
                  (None, 1): 3}
    for i in range(0, n):
        l, u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    x = [np.copy(np.array(x0[ib*n:(ib+1)*n], np.float64)) for ib in range(num_batches)]
    f = [np.copy(np.array(0.0, np.float64)) for ib in range(num_batches)]
    g = [np.copy(np.zeros((n,), np.float64)) for ib in range(num_batches)]
    wa = [np.copy(np.zeros(2*m*n + 5*n + 11*m*m + 8*m, np.float64)) for ib in range(num_batches)]
    iwa = [np.copy(np.zeros(3*n, np.int32)) for ib in range(num_batches)]
    task = [np.copy(np.zeros(1, 'S60')) for ib in range(num_batches)]
    csave = [np.copy(np.zeros(1, 'S60')) for ib in range(num_batches)]
    lsave = [np.copy(np.zeros(4, np.int32)) for ib in range(num_batches)]
    isave = [np.copy(np.zeros(44, np.int32)) for ib in range(num_batches)]
    dsave = [np.copy(np.zeros(29, np.float64)) for ib in range(num_batches)]
    for ib in range(num_batches):
        task[ib][:] = 'START'

    n_iterations = np.zeros(num_batches, dtype=np.int32)

    converged = num_batches * [False]

    warn_flag = np.zeros(num_batches)

    while not all(converged):
        pynvtx_range_push("LBFGS-ITERATION")
        for ib in range(num_batches):
            if converged[ib]:
                continue

            _lbfgsb.setulb(m, x[ib], low_bnd, upper_bnd, nbd, f[ib], g[ib], factr,
                           pgtol, wa[ib], iwa[ib], task[ib], iprint, csave[ib], lsave[ib],
                           isave[ib], dsave[ib], maxls)

        xk = np.concatenate(x)
        fk = func(xk)
        gk = fprime(xk)
        for ib in range(num_batches):
            if converged[ib]:
                continue
            task_str = task[ib].tostring()
            task_str_strip = task[ib].tostring().strip(b'\x00').strip()
            if task_str.startswith(b'FG'):
                # needs function evalation
                f[ib] = fk[ib]
                g[ib] = gk[ib*n:(ib+1)*n]
            elif task_str.startswith(b'NEW_X'):
                n_iterations[ib] += 1
                if n_iterations[ib] >= maxiter:
                    task[ib][:] = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            elif task_str_strip.startswith(b'CONV'):
                converged[ib] = True
                warn_flag[ib] = 0
            else:
                converged[ib] = True
                warn_flag[ib] = 2
                continue

        pynvtx_range_pop()
    xk = np.concatenate(x)

    if iprint > 0:
        print("CONVERGED in ({}-{}) iterations (|\/f|={})".format(np.min(n_iterations), np.max(n_iterations),
                                                                  np.linalg.norm(fprime(xk), np.inf)))

        if (warn_flag > 0).any():
            for ib in range(num_batches):
                if warn_flag[ib] > 0:
                    print("WARNING: id={} convergence issue: {}".format(ib, task[ib].tostring()))

    pynvtx_range_pop()
    return xk, n_iterations, warn_flag
