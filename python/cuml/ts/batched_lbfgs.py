import scipy.optimize as optimize
import numpy as np
from IPython.core.debugger import set_trace
from .batched_linesearch import batched_line_search_wolfe1
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


def Batched_I(r, num_batches):
    """Utility to build batched Identity Matrix (r x r) x num_batches: H = | I I I ... I |"""
    H = np.zeros((r, r * num_batches))
    for ib in range(num_batches):
        for ir in range(r):
            H[ir, ib*r + ir] = 1.0
    return H


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

    def func_nosum(x_in):
        return func(x_in, do_sum=False)

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
        fk = func_nosum(xk)
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


def batched_fmin_lbfgs(func, x0, num_batches, fprime=None, approx_grad=0, m=10,
                       factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1,
                       maxiter=50,
                       alpha_per_batch=True):

    if fprime is None and approx_grad is True:
        fprime = lambda x: _fd_fprime(x, func, epsilon)

    np.seterr(all='raise')

    pk = np.zeros(len(x0))
    sk = np.zeros(len(x0))
    yk = np.zeros(len(x0))

    N = len(x0) // num_batches
    Nb = len(x0)

    xkm1 = x0
    xk = x0
    yk = deque(maxlen=m)
    sk = deque(maxlen=m)
    rho = deque(maxlen=m)
    fk = []
    gk = np.zeros(Nb)

    k = 0

    is_converged = num_batches * [False]

    reset_LBFGS = False

    if iprint > 100:
        print("x0 = ", x0)

    for k_iter in range(maxiter):

        ######################
        # check convergence/stopping condition
        fk.append(func(xk))
        gkm1 = gk
        gk = fprime(xk)

        if (iprint > 0 and k % iprint == 0) or iprint > 100:
            # print("k:{} f={:0.5g}, |\/f|_inf={:0.5g}".format(k, fk[-1], np.linalg.norm(gk, np.inf)))
            print("k:{} f={}, |\/f|_inf={:0.5g}".format(k_iter, func(xk, do_sum=True),
                                                        np.linalg.norm(gk, np.inf)))
            if iprint > 100:
                print("xk = ", xk)

        if np.linalg.norm(gk, np.inf) < pgtol:
            print("CONVERGED: |g|_{inf} < PGTOL, STOPPING.")
            break

        if k > 1 and np.abs(fk[-2] - fk[-1]) < factr * np.finfo(float).eps:
            print("CONVERGED: Difference between last two iterations smaller than tolerance, stopping.")
            break

        # check individual series convergence
        for ib in range(num_batches):
            if np.linalg.norm(gk[ib*N:(ib+1)*N], np.inf) < pgtol:
                is_converged[ib] = True

        ######################
        # compute pk

        # gradient descent for first step
        if k == 0:
            pk = -gk
            for ib in range(num_batches):
                if is_converged[ib]:
                    pk[ib*N:(ib+1)*N] = 0.0
        # L-BFGS after first step
        else:

            sk.append(xk-xkm1)
            yk.append(gk-gkm1)
            rhok = np.zeros(num_batches)
            gamma_k = np.zeros(num_batches)

            # throw an error if divide by zero
            np.seterr(all='raise')

            for ib in range(num_batches):
                if is_converged[ib]:
                    continue
                rhok[ib] = 1/np.dot(yk[-1][ib*N:(ib+1)*N], sk[-1][ib*N:(ib+1)*N])
                gammaA = np.dot(sk[-1][ib*N:(ib+1)*N], yk[-1][ib*N:(ib+1)*N])
                gammaB = np.dot(yk[-1][ib*N:(ib+1)*N], yk[-1][ib*N:(ib+1)*N])
                gamma_k[ib] = gammaA / gammaB

            np.seterr(all='warn')

            rho.append(rhok)
            
            #############
            # Alg. 7.4

            pk = np.zeros(Nb)
            q = np.copy(gk)
            
            for ib in range(num_batches):
                if is_converged[ib]:
                    continue
                alpha_i = np.zeros(min(k, m))

                # first loop recursion
                for i in range(min(k, m)):
                    idx = -1-i
                    alpha_i[idx] = rho[idx][ib] * np.dot(sk[idx][ib*N:(ib+1)*N], q[ib*N:(ib+1)*N])
                    q[ib*N:(ib+1)*N] = q[ib*N:(ib+1)*N] - alpha_i[idx] * yk[idx][ib*N:(ib+1)*N]
                    
                H0k = gamma_k[ib] * np.eye(N)    
                r = H0k @ q[ib*N:(ib+1)*N]

                # second loop recursion
                for i in range(min(k, m)):
                    beta = rho[i][ib] * np.dot(yk[i][ib*N:(ib+1)*N], r)
                    r = r + sk[i][ib*N:(ib+1)*N] * (alpha_i[i] - beta)

                # note r = Hk \/f
                pk[ib*N:(ib+1)*N] = -r

        alpha_b = np.zeros(num_batches)
        xkp1 = np.copy(xk)

        alpha_batched = batched_line_search_wolfe1(func, fprime,
                                                   N, num_batches,
                                                   np.copy(xk), np.copy(pk),
                                                   is_converged)

        if iprint > 100:
            print("alpha = ", alpha_batched)
            print("pk = ", pk)
            print("gk = ", gk)
        
        for ib in range(num_batches):
            if alpha_batched[ib] > 0:
                xkp1[ib*N:(ib+1)*N] = xk[ib*N:(ib+1)*N] + alpha_batched[ib]*pk[ib*N:(ib+1)*N]
                alpha_b[ib] = alpha_batched[ib]
            if alpha_batched[ib] < 0:
                print("WARNING(k:{},ib:{}): Line search failed, resetting L-BFGS memory".format(k_iter,ib))
                reset_LBFGS = True

        if reset_LBFGS:
            # for now, we reset L-BFGS for every series. Eventually we
            # should only reset those who need it.
            k = 0
            yk.clear()
            sk.clear()
            rho.clear()
            reset_LBFGS = False
            # update state for those series who had line search success.
            xkm1 = np.copy(xk)
            xk = np.copy(xkp1)
            continue
        
        xkm1 = np.copy(xk)
        xk = np.copy(xkp1)
        k += 1

    print("CONVERGED: {} Iterations".format(k_iter))
    return xk, k_iter
