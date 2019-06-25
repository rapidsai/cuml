import scipy.optimize as optimize
import numpy as np
from IPython.core.debugger import set_trace
from .batched_linesearch import batched_line_search_wolfe1


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


def batched_fmin_lbfgs(func, x0, num_batches, fprime=None, approx_grad=0, m=10,
                       factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1,
                       maxiter=50,
                       alpha_per_batch=True):

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

    is_converged = num_batches * [False]

    reset_LBFGS = False

    for k_iter in range(maxiter):

        ######################
        # check convergence/stopping condition
        fk.append(func(xk))
        gkm1 = gk
        gk = fprime(xk)

        if iprint > 0 and k % iprint == 0:
            # print("k:{} f={:0.5g}, |\/f|_inf={:0.5g}".format(k, fk[-1], np.linalg.norm(gk, np.inf)))
            print("k:{} f={}, |\/f|_inf={:0.5g}".format(k_iter, func(xk, do_sum=True),
                                                        np.linalg.norm(gk, np.inf)))

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
