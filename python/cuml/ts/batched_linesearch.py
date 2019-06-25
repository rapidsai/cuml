import numpy as np
from typing import Tuple, List
from IPython.core.debugger import set_trace
from scipy.optimize import minpack2

def linesearch_minpack(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
                       stpmin,stpmax):

    print("Minpack linesearch")

def batched_line_search_armijo(f, nb, r,
                               xk, pk, gfk, fxk,
                               c1=1e-4,
                               alpha0=1,
                               alpha_min=0,
                               alpha_max=5) -> Tuple[float, int, np.ndarray]:
    """
    Backtracking (batched) line search to satisfy the armijo conditions.
    Returns:
    * alpha: vector of alpha>0 if converged, alpha[i] < 0 if search direction (suspected) not a descent direction
    * number of f evalutions
    * xk + alpha*pk: vector of solution, None if no solution found
    """

    def phi(alpha: np.ndarray) -> np.ndarray:
        xkp1 = np.copy(xk)
        for i in range(nb):
            xkp1[i*r:(i+1)*r] += alpha[i]*pk[i*r:(i+1)*r]
        return f(xkp1)

    if len(fxk) is not nb:
        raise ValueError("f(x) must return a vector of size `nb` (number of batches)")

    alpha_return = np.ones(nb)*-1

    # expand scalar alpha into nb-sized vector
    alpha0 = np.ones(nb)*alpha0

    phi0 = fxk
    phi_a0 = phi(alpha0)

    alpha_converged = nb*[False]

    derphi = np.zeros(nb)
    for ib in range(nb):
        derphi[ib] = np.dot(gfk[ib*r:(ib+1)*r], pk[ib*r:(ib+1)*r])

        # if search direction == 0, assume this series is converged.
        if derphi[ib] == 0:
            alpha_converged[ib] = True

    # check amijo condition for a quick return
    for ib in range(nb):
        if phi_a0[ib] <= phi0[ib] + c1*alpha0[ib]*derphi[ib]:
            alpha_converged[ib] = True
            alpha_return[ib] = alpha0[ib]
            
    if all(alpha_converged):
        return alpha0, 1, phi_a0
    
    # For the alpha-optimization routines below, see SciPy's `linesearch.py:scalar_search_armijo()`,
    # which, in turn, uses methods from Pgs, 56-58.

    # Compute a quadratic minimizer for alpha
    alpha1 = -derphi * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi * alpha0)
    for ib in range(nb):
        alpha1[ib] = max(alpha1[ib], alpha_max)
    
    for ib in range(nb):
        if alpha1[ib] < 0:
            # pk probably not a search direction.
            alpha_return[ib] = -1

    if np.any(alpha_return < 0):
        return alpha_return, 3, None

    phi_a1 = phi(alpha1)

    # check amijo condition for quadratic-optimized alpha
    for ib in range(nb):
        if alpha_converged[ib]:
            # skip already converged choices
            continue
        if phi_a1[ib] <= phi0[ib] + c1*alpha1[ib]*derphi[ib]:
            alpha_converged[ib] = True
            alpha_return[ib] = alpha1[ib]

    if all(alpha_converged):
        phi_return = phi(alpha_return)
        return alpha_return, 3, phi_return

    # If still not converged, loop using a cubic alpha-optimizer.
    fcall = 3
    while not all(alpha_converged):
        # recall: these should be pointwise array operations
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        alpha2 = np.zeros(nb)
        a = alpha0**2 * (phi_a1 - phi0 - derphi*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(np.abs(b**2 - 3 * a * derphi))) / (3.0*a)
        for ib in range(nb):
            alpha2[ib] = max(alpha2[ib], alpha_max)

        fcall += 1
        phi_a2 = phi(alpha2)

        for ib in range(nb):
            if alpha_converged[ib]:
                # skip already converged choices
                continue
            if phi_a2[ib] <= phi0[ib] + c1*alpha2[ib]*derphi[ib]:
                alpha_converged[ib] = True
                alpha_return[ib] = alpha1[ib]

        if all(alpha_converged):
            phi_return = phi(alpha_return)
            return alpha_return, fcall, phi_return

        if np.min(alpha2) < alpha_min:
            break

        if fcall > 10:
            break

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2


    # line search failed
    return alpha_return, fcall, None


def batched_line_search_wolfe1(f, fg, r: int, nb: int, x0: np.ndarray, pk: np.ndarray,
                               is_converged: List[bool],
                               amin=1e-8, amax=100, c1=1e-4, c2=0.9, xtol=1e-14,
                               max_ls_iter=10):

    # initial alpha = 1
    alphak = np.ones(nb)

    # Not found: 0
    # Found: alpha > 0
    # Line search failed: -1
    alpha_found = np.zeros(nb)
    
    # batched phi and phi'
    # phi(alpha) = f(xk + alpha*p)
    def phi(alpha: np.ndarray) -> np.ndarray:
        xkp1 = np.copy(x0)
        for i in range(nb):
            xkp1[i*r:(i+1)*r] += alpha[i]*pk[i*r:(i+1)*r]
        return f(xkp1, do_sum=False)

    # this returns one scalar per batch
    # phi'(alpha) = f'(xk + alpha*p).T * p (per batch member)
    def phip(alpha: np.ndarray) -> np.ndarray:
        xkp1 = np.copy(x0)
        for i in range(nb):
            xkp1[i*r:(i+1)*r] += alpha[i]*pk[i*r:(i+1)*r]
        gfp1 = fg(xkp1)
        gfp1_dot_pk = np.zeros(nb)
        for i in range(nb):
            gfp1_dot_pk[i] = np.dot(gfp1[i*r:(i+1)*r], pk[i*r:(i+1)*r])
        return gfp1_dot_pk
    

    k_ls = 0

    # minpack2.dcsrch state. One for each batch member.
    task = [b'START' for ib in range(nb)]
    isave = [np.copy(np.zeros((2,), np.intc)) for ib in range(nb)]
    dsave = [np.copy(np.zeros((13,), float)) for ib in range(nb)]

    phi1 = phi(np.zeros(nb))
    phip1 = phip(np.zeros(nb))

    while (alpha_found == 0.0).any() and k_ls < max_ls_iter:
        for ib in range(nb):

            # set to non-zero so we escape the while loop
            if is_converged[ib]:
                alpha_found[ib] = 1e-16

            # skip line search if:
            # alpha_found > 0 ("converged")
            # alpha_found < 0 ("failed")
            if alpha_found[ib] != 0.0 or is_converged[ib]:
                continue

            # print("input:", phi1[ib], phip1[ib],isave[ib],dsave[ib])
            alpha_ib, _, _, task_ib = minpack2.dcsrch(alphak[ib], phi1[ib], phip1[ib],
                                                      c1, c2, xtol,
                                                      task[ib],
                                                      amin, amax, isave[ib], dsave[ib])

            alphak[ib] = alpha_ib
            task[ib] = task_ib

            # line search failed
            if task_ib[:5] == b'ERROR' or task[:4] == b'WARN':
                alpha_found[ib] = -1
                alphak[ib] = 0.0
            # line search converged
            elif task_ib[:4] == b'CONV':
                alpha_found[ib] = alpha_ib
                alphak[ib] = 0.0

        # re-evaluate batched-function
        phi1 = phi(alphak)
        phip1 = phip(alphak)
        
        k_ls += 1

    if k_ls >= max_ls_iter:
        print("WARNING: Linesearch failed to converge under maximum number of line search iterations!")

    # reset alpha to 0.0 for already converged batch members
    for ib in range(nb):
        if is_converged[ib]:
            alpha_found[ib] = 0.0
    
    print("af:", alpha_found)
    return alpha_found
    
# def batched_line_search_wolfe2(f, gf, r: int, x0: np.ndarray, pk: np.ndarray, nb: int,
#                         amax=100, c1=1e-4, c2=0.9):
#     """
#     Incomplete version of a (batched) strong-wolfe conditions line-search.
#     """


#     # f returns a vector of size nb
#     phi0 = f(x0)
#     phip_0 = np.dot(pk, gf(x0))

#     if len(phi0) is not nb:
#         raise ValueError("f(x) must return a vector of size `nb` (number of batches)")

#     def phi(alpha: np.ndarray) -> np.ndarray:
#         xkp1 = np.copy(x0)
#         for i in range(nb):
#             xkp1[i*r:(i+1)*r] += alpha[i]*pk[i*r:(i+1)*r]
#         return f(xkp1)

#     def phip(alpha: np.ndarray) -> np.ndarray:
#         xkp1 = np.copy(x0)
#         for i in range(nb):
#             xkp1[i*r:(i+1)*r] += alpha[i]*pk[i*r:(i+1)*r]
#         gfp1 = gf(xkp1)
#         gfp1_dot_pk = np.zeros(nb)
#         for i in range(nb):
#             gfp1_dot_pk[i] = np.dot(gfp1[i*r:(i+1)*r], pk[i*r:(i+1)*r])
#         return gfp1_dot_pk

#     ############################################################
#     # Alg 3.5 Numerical Optimization
#     alpha_i = np.ones(nb)
#     alpha_i_m1 = np.ones(nb)
#     alpha_star = np.zeros(nb)
#     alpha_converged = nb*[False]
#     need_zoom = nb*[None]
#     phi_i_m1 = phi0
#     i = 1
#     while not all(alpha_converged):
#         phi_i = phi(alpha_i)
#         phip_i = phip(alpha_i)
#         for ib in range(nb):
#             # skip if alpha already good
#             if alpha_converged[ib] == True:
#                 continue

#             # first wolfe-condition failed; pick an alpha between alpha_{i-1} and alpha_i
#             if phi_i[ib] > phi0[ib] + c1*alpha_i[ib]*phip_0 or (i > 1 and phi_i >= phi_i_m1):
#                 need_zoom[ib] = (alpha_i_m1[ib], alpha_i[ib])

#             # second wolfe-condition: sufficient curvature condition
#             if abs(phip_i[ib]) =< -c2 * phip_0[ib]:
#                 # both conditions satisfied, use this alpha_i for batch 'i'
#                 alpha_converged[ib] = True
#                 alpha_star[ib] = alpha_i[ib]
#                 continue

#             # if step gradient is positive
#             if phip_i >= 0:
#                 need_zoom[ib] = (alpha_i[ib], amax)

        

# wolfe2 optimal result (slow way):
# k=000: 9.8236779 | (1.0000000, 2.0000000) | (0.0000000e+00,3.3045855e-02)
# k=001: 9.8232249 | (1.0000000, 1.0000000) | (0.0000000e+00,1.4072712e-02)
# k=002: 9.8223804 | (1.0000000, 16.0000000) | (0.0000000e+00,1.3861202e-02)
# /home/max/dev/cuml/python/external_builds/scipy/scipy/optimize/linesearch.py:466: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# /home/max/dev/cuml/python/external_builds/scipy/scipy/optimize/linesearch.py:314: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# k=003: 9.8222919 | (1.0000000, 1000.0000000) | (0.0000000e+00,1.5233375e-02)
# k=004: 9.8218456 | (1.0000000, 1.0000000) | (0.0000000e+00,1.5199005e-02)
# k=005: 9.8211148 | (0.3934023, 1.0000000) | (0.0000000e+00,2.6246457e-02)
# k=006: 9.8208148 | (1.0000000, 1.0000000) | (0.0000000e+00,4.0411044e-02)
# k=007: 9.8205507 | (1.0000000, 1.0000000) | (0.0000000e+00,3.9171725e-02)
# k=008: 9.8202554 | (0.4010477, 4.0000000) | (0.0000000e+00,4.1007900e-02)
# k=009: 9.8199542 | (1.0000000, 1.0000000) | (0.0000000e+00,4.2129566e-02)
# k=010: 9.8195398 | (0.0000000, 1.0000000) | (0.0000000e+00,4.9884066e-02)
# k=011: 9.8188804 | (0.0000000, 2.0000000) | (0.0000000e+00,5.2143261e-02)
# k=012: 9.8177865 | (0.0000000, 8.0000000) | (0.0000000e+00,5.6899601e-02)
# k=013: 9.8162097 | (0.0000000, 8.0000000) | (0.0000000e+00,5.4860509e-02)
# k=014: 9.8144118 | (0.0000000, 4.0000000) | (0.0000000e+00,3.2419256e-02)
# k=015: 9.8129764 | (0.0000000, 32.0000000) | (0.0000000e+00,1.8510464e-02)
# INFO(11): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [16:11] pk = [-0.00034   0.003062  0.      ], gk = [ 0.00034  -0.003062 -0.      ]
# INFO(19): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [16:19] pk = [-0.000295 -0.000457  0.      ], gk = [ 0.000295  0.000457 -0.      ]
# k=016: 9.8125586 | (0.0000000, 32.0000000) | (0.0000000e+00,1.3405793e-02)
# INFO(4): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [17:4] pk = [-0.000212 -0.00057   0.      ], gk = [ 0.000212  0.00057  -0.      ]
# INFO(5): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [17:5] pk = [-0.000249 -0.000343  0.      ], gk = [ 0.000249  0.000343 -0.      ]
# INFO(12): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [17:12] pk = [-0.000225  0.000391  0.      ], gk = [ 0.000225 -0.000391 -0.      ]
# k=017: 9.8124362 | (0.0000000, 8.0000000) | (0.0000000e+00,1.8997599e-02)
# INFO(37): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [18:37] pk = [-9.187483e-05  6.014827e-05  0.000000e+00], gk = [ 9.187483e-05 -6.014827e-05 -0.000000e+00]
# k=018: 9.8123768 | (0.0000000, 64.0000000) | (0.0000000e+00,2.4321052e-02)
# INFO(9): Caught invalid step (FloatingPointError=overflow encountered in exp), resetting H=I
# [19:9] pk = [-0.000118 -0.024321  0.      ], gk = [ 0.000118  0.024321 -0.      ]
# k=019: 9.8123294 | (0.0000000, 8.0000000) | (0.0000000e+00,1.8285735e-02)
# k=020: 9.8123021 | (0.0000000, 1.0000000) | (0.0000000e+00,1.2455979e-02)
# k=021: 9.8122622 | (0.0000000, 2.0000000) | (0.0000000e+00,1.3167844e-02)
# k=022: 9.8122333 | (0.0000000, 1.0000000) | (0.0000000e+00,2.0281864e-02)
# k=023: 9.8122045 | (0.0000000, 1.0000000) | (0.0000000e+00,2.3506076e-02)
# k=024: 9.8121701 | (0.0000000, 1.0000000) | (0.0000000e+00,1.6280352e-02)
# k=025: 9.8121331 | (0.0000000, 1.0000000) | (0.0000000e+00,1.6132294e-02)
# k=026: 9.8121030 | (0.0000000, 1.0000000) | (0.0000000e+00,1.9312221e-02)
# k=027: 9.8120600 | (0.0000000, 1.0000000) | (0.0000000e+00,1.8377610e-02)
# k=028: 9.8120016 | (0.0000000, 1.0000000) | (0.0000000e+00,1.8987685e-02)
# k=029: 9.8119455 | (0.0000000, 2.0000000) | (0.0000000e+00,1.1861106e-02)
# k=030: 9.8119143 | (0.0000000, 1.0000000) | (0.0000000e+00,1.0341536e-02)
# k=031: 9.8118996 | (0.0000000, 1.0000000) | (0.0000000e+00,6.3307705e-03)
# k=032: 9.8118948 | (0.0000000, 1.0000000) | (0.0000000e+00,1.6808467e-03)
# k=033: 9.8118933 | (0.0000000, 1.0000000) | (0.0000000e+00,9.0552888e-05)
# k=034: 9.8118931 | (0.0000000, 1.0000000) | (0.0000000e+00,2.1812010e-05)
# k=035: 9.8118931 | (0.0000000, 1.0000000) | (0.0000000e+00,9.9145498e-06)
# Stopping criterion reached |g|<pgtol: 3.079722963315992e-05 < 1e-05
# Final result: f(xk)=0, |\/f(xk)|=3.0797e-05, n_iter=36
# NITER= 36
