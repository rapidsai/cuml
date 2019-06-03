import scipy.optimize as optimize
import numpy as np


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


def batched_fmin_bfgs(f, x0, num_batches, g=None, h=1e-8, pgtol=1e-5, factr=1e7, max_steps=100, disp=0):
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
    r = x0//num_batches

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

    fk = np.zeros(max_steps)
    xk = x0
    fk[0] = f(xk)

    if disp > 0:
        print("step | f(xk) | |\/f(xk)|")

    for k in range(max_steps):

        
        # 1. compute search direction ($p_k = -H_k \grad f_k$)
        gk = g(xk)

        if np.linalg.norm(gk) < pgtol:
            if(disp > 0):
                print("Stopping criterion reached |g|<pgtol: {} < {}".format(np.linalg.norm(gk), pgtol))
            break

        for ib in range(num_batches):
            pk[ib*r:(ib+1)*r] = - Hk[:, ib*r:(ib+1)*r] @ gk[ib*r:(ib+1)*r]

        # 2. set next step via linesearch
        alpha, fc, gc, fkp1, fk, gkp1 = optimize.line_search(f, g, xk, pk, gfk=gk)
        
        if fkp1 is None:
            raise ValueError("Line search failed to converge")
        xkp1 = xk + alpha * pk

        # 3. get BFGS variables `s_k` and `y_k`
        ## $s_k = x_{k+1}-x_{k},
        sk = xkp1 - xk
        ## y_k=\grad f_{k+1} - \grad f_{k}$
        yk = gkp1 - gk

        # 4. Compute $H_{k+1}$ by BFGS update (eq. 6.17 in N&W)
        Hkp1 = np.zeros((r, r * num_batches))
        # eq. (6.14)
        rhok = 1/np.dot(yk, sk)

        for ib in range(num_batches):
            Ib = np.eye(r)
            I_m_rho_y_s = (Ib - rhok * sk[ib*r:(ib+1)*r] @ yk[ib*r:(ib+1)*r].T)
            Hk_batch = Hk[:, ib*r:(ib+1)*r]
            rho_s_sT = rhok * sk[ib*r:(ib+1)*r] @ sk[ib*r:(ib+1)*r].T
            Hkp1_batch = I_m_rho_y_s@ Hk_batch @ I_m_rho_y_s + rho_s_sT
            Hkp1[:, ib*r:(ib+1)*r] = Hkp1_batch

        # print diagnostic information
        if disp > 0 and disp < 100:
            if k % disp == 0:
                print("k={}: {} | {}".format(k, f(xk), g(xk)))
        if disp > 100:
            print("k={}: {} | {}".format(k, f(xk), g(xk)))
            print("Line Search: fc={}, gc={}, alpha={}, |alpha*p|={}".format(fc, gc, alpha,
                                                                             np.linalg.norm(alpha*pk)))

        Hk = Hkp1
        xk = xkp1
        fk[k+1] = fkp1

        # stopping criterion: f(x) in last steps not changed
        num_steps = 5
        if k>num_steps:
            if np.mean(np.abs(fkp1 - fk[k-num_steps:k])) < factr*1e-22:
                print("Stopping criterion true: Last {} steps almost no change in f(x)".format(num_steps))

        if k==max_steps-1:
            print("Stopping criterion: Maximum number of iterations!")

    if disp > 0:
        print("Final result: f(xk)={}, |\/f(xk)|={}, n_iter={}".format(fk, gk, k))

    return xk, fk



