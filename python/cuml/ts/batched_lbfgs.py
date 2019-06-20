import scipy.optimize as optimize
import numpy as np
from IPython.core.debugger import set_trace

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


def batched_fmin_lbfgs(func, x0, fprime=None, args=(), approx_grad=0, bounds=None, m=10,
                       factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1,
                       maxfun=15000, maxiter=1000, disp=None, callback=None, maxls=20):


    if fprime is None and approx_grad is True:
        fprime = lambda x: _fd_fprime(x, func, epsilon)

    pk = np.zeros(len(x0))
    sk = np.zeros(len(x0))
    yk = np.zeros(len(x0))

    N = len(x0)
    y_k_m_1 = np.ones(N)
    s_k_m_1 = np.ones(N)

    xkm1 = x0
    xk = x0
    yk = deque(maxlen=m)
    sk = deque(maxlen=m)
    pk = deque(maxlen=m)
    rho = deque(maxlen=m)
    fk = []
    gk = np.zeros(N)

    Hk_bfgs = np.eye(N)

    k = 0

    for k_iter in range(maxiter):

        ######################
        # check convergence/stopping condition
        fk.append(func(xk))
        gkm1 = gk
        gk = fprime(xk)

        if iprint > 0 and k % iprint == 0:
            print("k:{} f={:0.5g}, |\/f|_inf={:0.5g}".format(k, fk[-1], np.linalg.norm(gk, np.inf)))

        if np.linalg.norm(gk, np.inf) < pgtol:
            print("INFO: |g|_{inf} < PGTOL, STOPPING.")
            break

        if k > 1 and np.abs(fk[-2] - fk[-1]) < 1e-10:
            print("INFO: Last two iterations essentially identical, stopping.")
            break

        ######################
        # compute pk

        # gradient descent for first step
        if k == 0:
            pk = -fprime(x0)
            
        # L-BFGS after first step
        else:

            sk.append(xk-xkm1)
            yk.append(gk-gkm1)
            rho.append(1/np.dot(yk[-1], sk[-1]))

            gamma_k = np.dot(sk[-1], yk[-1]) / np.dot(yk[-1], yk[-1])

            #############
            # Alg. 7.4
            # first loop recursion
            alpha_i = np.zeros(min(k, m))
            q = gk
            for i in range(min(k, m)):
                idx = -1-i
                alpha_i[idx] = rho[idx] * np.dot(sk[idx], q)
                q = q - alpha_i[idx] * yk[idx]

            
            H0k = gamma_k * np.eye(N)
            r = H0k @ q
            # second loop recursion
            for i in range(min(k, m)):
                beta = rho[i] * np.dot(yk[i], r)
                r = r + sk[i] * (alpha_i[i] - beta)


            # note r = Hk \/f
            pk = -r

        # alpha, fc, gc, fkp1, _, _ = optimize.line_search(func, fprime,
        #                                                  xk, pk)
        alpha, fc, gc, fkp1, _, gkp1 = optimize.linesearch.line_search_wolfe1(func,
                                                                              fprime, xk,
                                                                              pk, gk)

        if alpha is None:
            # Line-Search failed, reset L-BFGS memory.
            k = 0
            yk.clear()
            sk.clear()
            pk.clear()
            rho.clear()
            continue

        xkp1 = xk + alpha*pk

        xkm1 = xk
        xk = xkp1
        k += 1


    return xk, k
