import numpy as np
from typing import Tuple
from IPython.core.debugger import set_trace

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

        

