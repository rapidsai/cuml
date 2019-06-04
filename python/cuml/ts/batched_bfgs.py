import scipy.optimize as optimize
import numpy as np
from IPython.core.debugger import set_trace

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


def backtracking_line_search(f, g, xk, pk, gfk=None):
    """Backtracking line search from pg 37 of N&W. Same API as scipy.optimize.line_search"""
    if gfk is None:
        gfk = g(xk)
    fxk0 = f(xk)
    alpha = 1.0
    max_num_ls_iter = 10
    c = 0.01
    # set_trace()
    for k in range(max_num_ls_iter):
        lhs = f(xk + alpha*pk)
        rhs = fxk0 + c*alpha*gfk.dot(pk)
        if lhs < rhs:
            break
        if k == max_num_ls_iter-1:
            print("Maximum number of line search iteratios")

        alpha *= 0.5

    return alpha, k, 1, f(xk+alpha*pk), c, g(xk+alpha*pk)
    

def batched_fmin_bfgs(f, x0, num_batches, g=None, h=1e-8,
                      pgtol=1e-5, factr=1e7, max_steps=100, disp=0,
                      alpha_per_batch=True):
    """
    Batched minimizer using BFGS algorithm.
    batchsize: Assume that f, g, and x0 are already batched, and `x` and `g` can be grouped by batches.
    h: finite difference stepsize when gradient `g=None`
    pgtol: Gradient stopping criterion. Stop when ||g||_\inf < pgtol.
    factr: When objective `f` stops changing below 1e-12*factr, i.e., |f(x_{k-1}) - f(x_{k})| < 1e-12 * factr.
    max_steps: Maximum number of optimization steps
    disp: Debug info --- 1-100 display modulus number, >100, display detailed information every step

    Return:
    xk: Minimized parameters
    fxk, Function value at final minimized parameter
    """

    # finite differencing if no gradient function provided
    if g is None:
        g = lambda x: _fd_fprime(x, f, h)

    if len(x0) % num_batches is not 0:
        raise ValueError("The number of DOF in x0 does not divide into the number of batches evenly")

    # number of variables/parameters in each batch
    r = len(x0)//num_batches

    # Batch notes:
    # * All vectors are flat with implicit batching every "num_batches".
    # * Hessian matrix H is a (r x r * num_batches matrix) i.e., Horizontally
    #   concatenated blocks of size `r x r`.

    # BFGS from Algorithm 6.1, pg.140 of "Numerical Optimization" by Nocedal and Wright.

    # 0. Init H_0 with Identity
    Hk = np.zeros((r, r * num_batches))
    for ib in range(num_batches):
        for ir in range(r):
            Hk[ir, ib*r + ir] = 1.0

    pk = np.zeros(len(x0))
    sk = np.zeros(len(x0))
    yk = np.zeros(len(x0))

    fk = np.zeros(max_steps+1)
    xk = x0
    fk[0] = f(xk)
    fkm1 = fk[0] + np.linalg.norm(g(xk)) / 2

    if disp > 0:
        print("step   f(xk)     | alpha  | \/f(xk)")

    for k in range(max_steps):

        gk = g(xk)
        if np.linalg.norm(gk) < pgtol:
            if(disp > 0):
                print("Stopping criterion reached |g|<pgtol: {} < {}".format(np.linalg.norm(gk), pgtol))
            break

        xkp1 = np.zeros(len(xk))

        # 1. compute search direction ($p_k = -H_k \grad f_k$)
        for ib in range(num_batches):
            pk[ib*r:(ib+1)*r] = - Hk[:, ib*r:(ib+1)*r] @ gk[ib*r:(ib+1)*r]

        if np.isnan(pk).any():
            raise ValueError("pk NaN")

        # 2. set next step via linesearch
        if alpha_per_batch:
            # compute alpha per batch
            alpha_b = np.zeros(num_batches)
            xkp1 = np.zeros(len(xk))

            for ib in range(num_batches):
                # When we are too close to minimum, line search fails. Don't
                # search if we are more than satisfying the stopping criterion.
                if(np.linalg.norm(gk[r*ib:r*(ib+1)]) > 1e-2*pgtol):
                    alpha, fc, gc, fkp1, _, _ = optimize.line_search(f, g,
                                                                     xk[ib*r:(ib+1)*r],
                                                                     pk[ib*r:(ib+1)*r],
                                                                     args=(ib,))
                    alpha_b[ib] = alpha
                    if fkp1 is None:
                        print("bid({})|gk|={},|pk|={}".format(ib, np.linalg.norm(gk[ib*r:(ib+1)*r]),
                                                   np.linalg.norm(pk[ib*r:(ib+1)*r])))
                        raise ValueError("Line search failed to converge")
                
                xkp1[ib*r:(ib+1)*r] = xk[ib*r:(ib+1)*r] + alpha_b[ib] * pk[ib*r:(ib+1)*r]

        else:
            # compute alpha for the global optimization problem
            alpha, fc, gc, fkp1, _, _ = optimize.line_search(f, g,
                                                             xk,
                                                             pk)
            
            if fkp1 is None:
                    raise ValueError("Line search failed to converge")
            xkp1 = xk + alpha*pk

        gkp1 = g(xkp1)

        # 3. get BFGS variables `s_k` and `y_k`
        ## $s_k = x_{k+1}-x_{k},
        sk = xkp1 - xk
        ## y_k=\grad f_{k+1} - \grad f_{k}$
        yk = gkp1 - gk

        # 4. Compute $H_{k+1}$ by BFGS update (eq. 6.17 in N&W)
        Hkp1 = np.zeros((r, r * num_batches))
        
        # Run BFGS update on each batch seperately
        for ib in range(num_batches):
            # eq. (6.14)
            sk_dot_yk = np.dot(yk[ib*r:(ib+1)*r], sk[ib*r:(ib+1)*r])
            if sk_dot_yk == 0:
                continue
            rhok = 1/sk_dot_yk
            if np.isnan(rhok) or np.isinf(rhok):
                raise ValueError("NaN or Inf rho_k")
            Ib = np.eye(r)
            A1 = Ib - rhok * np.outer(sk[ib*r:(ib+1)*r], yk[ib*r:(ib+1)*r])
            A2 = Ib - rhok * np.outer(yk[ib*r:(ib+1)*r], sk[ib*r:(ib+1)*r])
            Hk_batch = Hk[:, ib*r:(ib+1)*r]
            rho_s_sT = rhok * np.outer(sk[ib*r:(ib+1)*r], sk[ib*r:(ib+1)*r])
            Hkp1_batch = A1@ Hk_batch @ A2 + rho_s_sT
            Hkp1[:, ib*r:(ib+1)*r] = Hkp1_batch

        # print diagnostic information
        if disp > 0 and disp < 100:
            if k % disp == 0:
                disp_amt = min(r, 4)
                print("k={:03d}: {:0.7f} | {:0.4f} | {}".format(k, f(xk), alpha,
                                                                g(xkp1)[:disp_amt]))
        if disp > 100:
            print("k={:03d}: {:0.7f} | {:0.4f} | {}".format(k, f(xk), alpha, g(xk)))
            print("Line Search: fc={}, gc={}, alpha={:0.4f}, |alpha*p|={:0.5f}".format(fc, gc, alpha,
                                                                             np.linalg.norm(alpha*pk)))

        Hk = Hkp1
        xk = xkp1
        fk[k+1] = fkp1

        # stopping criterion: f(x) in last steps not changed
        num_steps = 5
        if k>num_steps:
            if np.mean(np.abs(fkp1 - fk[k-num_steps:k])) < factr*1e-22:
                if disp > 0:
                    print("Stopping criterion true: Last {} steps almost no change in f(x)".format(num_steps))
                break

        if k==max_steps-1:
            if disp > 0:
                print("Stopping criterion: Maximum number of iterations!")

    if disp > 0:
        print("Final result: f(xk)={}, |\/f(xk)|={}, n_iter={}".format(fk[-1],
                                                                       np.linalg.norm(gk),
                                                                       k))

    return xk, fk
