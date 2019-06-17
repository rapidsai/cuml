import scipy.optimize as optimize
import numpy as np
from IPython.core.debugger import set_trace

from .batched_linesearch import batched_line_search_armijo

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
                      alpha_per_batch=False, alpha_max=100):
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
    
    def Batched_I():
        """Utility to build batched Identity Matrix (r x r) x num_batches: H = | I I I ... I |"""
        H = np.zeros((r, r * num_batches))
        for ib in range(num_batches):
            for ir in range(r):
                H[ir, ib*r + ir] = 1.0
        return H

    Hk = Batched_I()

    pk = np.zeros(len(x0))
    sk = np.zeros(len(x0))
    yk = np.zeros(len(x0))

    fk = np.zeros(max_steps+1)
    xk = x0
    fk[0] = f(xk)
    fkm1 = fk[0] + np.linalg.norm(g(xk)) / 2

    if disp > 0:
        print("step   f(xk)     | alpha  | \/f(xk)")

    k = 0
    k_ls_reset = 0
    while True:
        if k >= max_steps:
            if disp > 0:
                print("Stopping criterion: Maximum number of iterations!")
            break

        gk = g(xk)
        if np.linalg.norm(gk, ord=np.inf) < pgtol:
            if(disp > 0):
                print("Stopping criterion reached |g|<pgtol: {} < {}".format(np.linalg.norm(gk), pgtol))
            break

        xkp1 = np.zeros(len(xk))

        # 1. compute search direction ($p_k = -H_k \grad f_k$)
        # for ib in range(num_batches):
        #     pk[ib*r:(ib+1)*r] = - Hk[:, ib*r:(ib+1)*r] @ gk[ib*r:(ib+1)*r]

        if np.isnan(pk).any():
            raise FloatingPointError("pk NaN")


        # 2. set next step via linesearch
        if alpha_per_batch:
            # compute alpha per batch
            alpha_b = np.zeros(num_batches)
            xkp1 = np.zeros(len(xk))

            ls_option = 3

            if ls_option == 3:
                ls_iterations = 0
                while True:

                    # Bail out if more than 5 line search iterations
                    ls_iterations += 1
                    if ls_iterations > 5:
                        raise ValueError("ERROR: Too many line search iterations")

                    pk = np.zeros(len(x0))
                    for ib in range(num_batches):
                        if(np.linalg.norm(gk[r*ib:r*(ib+1)]) > pgtol):
                            pk[ib*r:(ib+1)*r] = - Hk[:, ib*r:(ib+1)*r] @ gk[ib*r:(ib+1)*r]

                    
                    # home-made, batch aware line search satisfying Armijo conditions
                    def f2(x):
                        return f(x, do_sum=False)
                    try:
                        alpha_b, fc, fkp1 = batched_line_search_armijo(f2, num_batches, r,
                                                                     xk, pk, gk, f2(xk))

                    # catch errors in transform
                    except FloatingPointError as fpe:
                        print("INFO: Caught invalid step (FloatingPointError={}), resetting H=I".format(fpe))
                        for ib in range(num_batches):
                            Hk[:, ib*r:(ib+1)*r] = np.eye(r)
                        continue
                    
                    ##################################
                    # check return and possibly restart line search
                    restart_ls = False
                    for ib in range(num_batches):
                        # if any alpha < 0, reset that series H=I, and restart line-search
                        if alpha_b[ib] < 0:
                            Hk[:, ib*r:(ib+1)*r] = np.eye(r)
                            restart_ls = True

                    if restart_ls:
                        # restart line search
                        print("INFO: Restarting LS with some H=I")
                        continue

                    ##################################
                    # apply alpha
                    for ib in range(num_batches):
                        xkp1[ib*r:(ib+1)*r] = xk[ib*r:(ib+1)*r] + alpha_b[ib] * pk[ib*r:(ib+1)*r]

                    # line search successful, break
                    break

                    

            else:
                for ib in range(num_batches):
                    # When we are too close to minimum, line search fails. Don't
                    # search if we are satisfying the stopping criterion.
                    if(np.linalg.norm(gk[r*ib:r*(ib+1)]) > pgtol):
                        line_search = True
                        line_search_iterations = 0

                        while line_search:
                            pk = np.zeros(len(x0))
                            pk[ib*r:(ib+1)*r] = - Hk[:, ib*r:(ib+1)*r] @ gk[ib*r:(ib+1)*r]
                            try:
                                if line_search_iterations > 0:
                                    print("[{}:{}] pk = {}, gk = {}".format(k, ib, pk[ib*r:(ib+1)*r], gk[ib*r:(ib+1)*r]))

                                if ls_option == 1:
                                    # line-search to satisfy strong wolfe conditions
                                    alpha, fc, gc, fkp1, _, _ = optimize.line_search(f, g,
                                                                                     xk, pk,
                                                                                     # xk[ib*r:(ib+1)*r],
                                                                                     # pk[ib*r:(ib+1)*r],
                                                                                     args=(ib,), amax=alpha_max)
                                elif ls_option == 2:
                                    # line-search to satisfy armijo conditions
                                    gc = 1
                                    alpha, fc, fkp1 = optimize.linesearch.line_search_armijo(f, xk,
                                                                                             pk, gk,
                                                                                             f(xk),
                                                                                             args=(ib,))

                                if alpha is None or fkp1 is None:
                                    print("bid({})|gk|={},|pk|={}".format(ib, np.linalg.norm(gk[ib*r:(ib+1)*r]),
                                                                          np.linalg.norm(pk[ib*r:(ib+1)*r])))
                                    print("alpha={}, fkp1={}".format(alpha, fkp1))
                                    print("INFO: Line search failed: Resetting H=I")
                                    Hk[:, ib*r:(ib+1)*r] = np.eye(r)
                                    line_search_iterations += 1
                                    if line_search_iterations > 5:
                                        # raise ValueError("Line search failed to converge after 5 tries")
                                        print("INFO: Line search failed to converge after 5 tries, setting alpha=1")
                                        alpha_b[ib] = 1
                                        break
                                    continue
                                else:
                                    alpha_b[ib] = alpha
                                    break

                            except FloatingPointError as fpe:
                                # Reset H to identity to force pk to be gradient descent
                                # set_trace()
                                line_search_iterations += 1
                                if line_search_iterations > 5:
                                    raise ValueError("Line search failed to converge after 5 tries")
                                print("INFO({}): Caught invalid step (FloatingPointError={}), resetting H=I".format(ib, fpe))
                                Hk[:, ib*r:(ib+1)*r] = np.eye(r)
                                continue

                    # print("[{}:{}] xkp1 = {} + ({}) * {} (gk={})".format(k, ib, xk[ib*r:(ib+1)*r], alpha_b[ib], pk[ib*r:(ib+1)*r], gk[ib*r:(ib+1)*r]))
                    # print("pk = -{} @ {}".format(Hk[:, ib*r:(ib+1)*r], gk[ib*r:(ib+1)*r]))
                    xkp1[ib*r:(ib+1)*r] = xk[ib*r:(ib+1)*r] + alpha_b[ib] * pk[ib*r:(ib+1)*r]

        else:
            ls_option = 1
            
            # compute alpha for the global optimization problem
            if ls_option == 1:
                # set_trace()
                try:
                    alpha, fc, gc, fkp1, fkm1, _ = optimize.line_search(f, g,
                                                                        xk,
                                                                        pk, amax=alpha_max)
                except FloatingPointError as fpe:
                    # Reset H to identity to force pk to be gradient descent
                    print("INFO: Caught invalid step (FloatingPointError={}), resetting H=I".format(fpe))
                    Hk = Batched_I()
                    
                    continue

                if fkp1 is None or alpha is None:
                    # set_trace()
                    if disp > 0:
                        print("INFO: Line search failed to converge. Resetting H=I")
                    k_ls_reset += 1
                    if k_ls_reset > 3:
                        print("WARNING: Line search reset failed.")
                        break

                    # Reset H to identity to force pk to be gradient descent
                    Hk = Batched_I()
                    
                    continue

            if ls_option == 2:
                try:
                    alpha, fc, gc, fkp1, fkm1, _ = optimize.optimize._line_search_wolfe12(f, g, xk, pk, gk,
                                                                                          fk[k], fkm1, amax=alpha_max)
                except optimize.optimize._LineSearchError:
                    if disp > 0:
                        print("Warning: Line search failed to converge")
                    break

            # take step along search direction with line-search-computed alpha stepsize
            xkp1 = xk + alpha*pk
            k_ls_reset = 0

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
                if disp > 100:
                    print("batch({}) converged: sk_dot_yk==0".format(ib))
                continue
            rhok = 1/sk_dot_yk
            if np.isnan(rhok) or np.isinf(rhok):
                raise ValueError("NaN or Inf rho_k")
            Ib = np.eye(r)
            A1 = Ib - np.outer(sk[ib*r:(ib+1)*r], yk[ib*r:(ib+1)*r]) * rhok
            A2 = Ib - np.outer(yk[ib*r:(ib+1)*r], sk[ib*r:(ib+1)*r]) * rhok
            Hk_batch = Hk[:, ib*r:(ib+1)*r]
            rho_s_sT = rhok * np.outer(sk[ib*r:(ib+1)*r], sk[ib*r:(ib+1)*r])
            Hkp1_batch = A1@ Hk_batch @ A2 + rho_s_sT
            Hkp1[:, ib*r:(ib+1)*r] = Hkp1_batch

        # print diagnostic information
        if disp > 0 and disp < 100:
            if k % disp == 0:
                disp_amt = min(r, 4)
                if isinstance(alpha_b, np.ndarray):
                    print("k={:03d}: {:0.7f} | ({:0.7f}, {:0.7f}) | {}".format(k, f(xk), np.min(alpha_b), np.max(alpha_b),
                                                                g(xkp1)[:disp_amt]))
                else:
                    print("k={:03d}: {:0.7f} | {} | {}".format(k, f(xk), alpha,
                                                               g(xkp1)[:disp_amt]))
        if disp > 100:
            print("k={:03d}: {:0.7f} | {:0.4f} | {}".format(k, f(xk), alpha, g(xk)))
            print("Line Search: fc={}, gc={}, alpha={:0.4f}, |alpha*p|={:0.5f}".format(fc, gc, alpha,
                                                                             np.linalg.norm(alpha*pk)))

        Hk = Hkp1
        xk = xkp1
        if isinstance(fkp1, np.ndarray):
            fk[k+1] = fkp1.sum()
        else:
            fk[k+1] = fkp1
        

        # stopping criterion: f(x) in last steps not changed
        num_steps = 5
        if k>num_steps:
            if np.mean(np.abs(fk[k+1] - fk[k-num_steps:k])) < factr*1e-22:
                if disp > 0:
                    print("Stopping criterion true: Last {} steps almost no change in f(x)".format(num_steps))
                break

        
        k += 1
        # end while

    if disp > 0:
        print("Final result: f(xk)={:0.5g}, |\/f(xk)|={:0.5g}, n_iter={}".format(fk[-1],
                                                                       np.linalg.norm(gk),
                                                                       k))

    return xk, fk, k
