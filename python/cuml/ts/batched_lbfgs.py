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


def Batched_I(r, num_batches):
    """Utility to build batched Identity Matrix (r x r) x num_batches: H = | I I I ... I |"""
    H = np.zeros((r, r * num_batches))
    for ib in range(num_batches):
        for ir in range(r):
            H[ir, ib*r + ir] = 1.0
    return H


def batched_fmin_lbfgs(func, x0, num_batches, fprime=None, args=(), approx_grad=0, bounds=None, m=10,
                       factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1,
                       maxfun=15000, maxiter=50, disp=None, callback=None, maxls=20):


    if fprime is None and approx_grad is True:
        fprime = lambda x: _fd_fprime(x, func, epsilon)

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

        if k > 1 and np.abs(fk[-2] - fk[-1]) < factr * np.finfo(float).eps:
            print("INFO: Difference between last two iterations smaller than tolerance, stopping.")
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
            rhok = np.zeros(num_batches)
            gamma_k = np.zeros(num_batches)
            for ib in range(num_batches):
                rhok[ib] = 1/np.dot(yk[-1][ib*N:(ib+1)*N], sk[-1][ib*N:(ib+1)*N])
                gammaA = np.dot(sk[-1][ib*N:(ib+1)*N], yk[-1][ib*N:(ib+1)*N])
                gammaB = np.dot(yk[-1][ib*N:(ib+1)*N], yk[-1][ib*N:(ib+1)*N])
                gamma_k[ib] = gammaA / gammaB

            rho.append(rhok)
            # gamma_k = np.dot(sk[-1], yk[-1]) / np.dot(yk[-1], yk[-1])
            # rho.append(1/np.dot(yk[-1], sk[-1]))
            # set_trace()
            #############
            # Alg. 7.4

            pk = np.zeros(Nb)
            q = np.copy(gk)
            
            for ib in range(num_batches):
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

        alpha, _, _, _, _, _ = optimize.linesearch.line_search_wolfe1(func,
                                                                      fprime, xk,
                                                                      pk, gk)

        if alpha is None:
            # Line-Search failed, reset L-BFGS memory.
            print("WARNING: Line search failed, resetting L-BFGS memory")
            k = 0
            yk.clear()
            sk.clear()
            rho.clear()
            continue

        xkp1 = xk + alpha*pk

        xkm1 = xk
        xk = xkp1
        k += 1

    return xk, k
