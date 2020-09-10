#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
from cuml.sparse.linalg.robust_lobpcg import lobpcg as robust_lobpcg
from cuml.sparse.linalg.blopex_lobpcg import lobpcg as blopex_lobpcg

def lobpcg(A,
           X=None,
           B=None,
           M=None,
           k=None,
           maxiter=None,
           method= 'blopex',
           tol=None,
           largest=True,
           verbosityLevel=0,
           ortho_iparams=None,
           ortho_fparams=None,
           ortho_bparams=None,
           retLambdaHistory=False,
           retResidualNormsHistory=False
           ):

    """
    Find the k largest (or smallest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive defined generalized
    eigenvalue problem using matrix-free LOBPCG methods.
    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:
      `method="blopex"` - the LOBPCG method implemented by Andrew Knyazev,
      see [Knyazev2007]. It is also the algorithm implemented by Scipy.
      It is the default and most accurate.
      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. A less robust method, may fail when
      Cholesky is applied to singular input.
      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. A robust method.
    Supported input is only dense input for now.
    .. note:: In general, the basic and blopex methods spends least
      time per
      iteration. However, the robust method converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.


    Parameters
    ----------

    A : cupy.ndarray
            input ndarray of shape (n,n)

    B : cupy.ndarray, optional
            When not specified, `B` is interpereted as
            identity matrix.

    X : cupy.ndarray, optional
            the input ndarray of shape (n,k)
            When specified, it is used as
            initial approximation of eigenvectors. X must be a
            dense ndarray.

    M : cupy.ndarray, optional
            the input ndarray of size (n,n).
            When specified, it will be used as preconditioner.

    k : integer, optional (but needs to be specified if X isn't)
            the number of requested
            eigenpairs.

    maxiter : int, optional
            maximum number of iterations.
            When reached, the iteration process is hard-stopped and
            the current approximation of eigenpairs is returned.
            For infinite iteration but until convergence criteria
            is met, use `-1`.

    method : str, optional
            select LOBPCG method.
            `method="ortho"` - LOBPCG method with
                orthogonal basis selection [StathopoulosEtal2002].
                A robust method.
            `method="blopex"` - the LOBPCG method implemented by
                Andrew Knyazev, see [Knyazev2007]. It is also the
                algorithm implemented by Scipy.
            `method="basic"` - the LOBPCG method introduced by Andrew
                Knyazev, see [Knyazev2001]. A less robust method,
                may fail when Cholesky is applied to singular input.

    tol : float, optional
            residual tolerance for stopping
            criterion. Default is `feps ** 0.5` where `feps` is
            smallest non-zero floating-point number of the given
            input ndarray `A` data type.

    largest : bool, optional
            when True, solve the eigenproblem for
            the largest eigenvalues. Otherwise, solve the
            eigenproblem for smallest eigenvalues. Default is
            `True`.

    VerbosityLevel : int, optional
            Prints verbose trace of the iterations if value
            greater than 0.

    ortho_iparams, ortho_fparams, ortho_bparams : dict, optional
            various parameters to LOBPCG algorithm when using
            `method="ortho"`.

    retLambdaHistory: boolean, optional
            Whether to return the history of eigen-values as a list
            of arrays

    retResidualNormsHistory: boolean, optional
            Whether to return the residual L2 norm history of
            eigen-vectors as a list of ndarrays.

    Returns
    -------

    E : cupy.ndarray
            eigenvalues of size (k,)
    X : cupy.ndarray
            eigenvectors of size (n,k)
    lambdas : list of cupy.ndarrays, optional
            The eigenvalue history, if `retLambdaHistory` is True.
    rnorms : list of cupy.ndarrays
            The history of residual norms, if `retResidualNormsHistory`
            is True.

    References
    ----------

    [Kyanzev2007] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov
    (2007), Block Locally Optimal Preconditioned Eigenvalue Xolvers
    (BLOPEX) in hypre and PETSc. https://arxiv.org/abs/0705.2626
    [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal
    Preconditioned Eigensolver: Locally Optimal Block Preconditioned
    Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),
    517-541. (25 pages)
    https://epubs.siam.org/doi/abs/10.1137/S1064827500366124
    [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng
    Wu. (2002) A Block Orthogonalization Procedure with Constant
    Synchronization Requirements. SIAM J. Sci. Comput., 23(6),
    2165-2182. (18 pages)
    https://epubs.siam.org/doi/10.1137/S1064827500370883
    [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming
    Gu. (2018) A Robust and Efficient Implementation of LOBPCG.
    SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)
    https://epubs.siam.org/doi/abs/10.1137/17M1129830

    Notes
    -----

    If both ``retLambdaHistory`` and ``retResidualNormsHistory`` are True,
    the return tuple has the following format
    ``(lambda, V, lambda history, residual norms history)``.
    In the following ``n`` denotes the matrix size and ``k`` the number
    of required eigenvalues (smallest or largest).
    The LOBPCG code internally solves eigenproblems of the size ``3k`` on every  iteration by calling the "standard" dense eigensolver, so if ``k`` is not  small enough compared to ``n``, it does not make sense to call the LOBPCG code, but rather one should use the "standard" eigensolver.
    In the BLOPEX implementation, (i.e, method="blopex"),
    if one calls the LOBPCG algorithm for ``5k > n``, it will most likely break internally, so the code tries to call the standard function instead.
    It is not that ``n`` should be large for the LOBPCG to work, but rather the ratio ``n / k`` should be large. It you call LOBPCG with ``k=1`` and ``n=10``, it works though ``n`` is small. The method is intended for extremely large ``n / k``

    In the other 2 implementations, if ``3k > n``, it Raises an Error for the same reason as stated above.

    The convergence speed depends basically on two factors:
    1. How well relatively separated the seeking eigenvalues
    are from the rest
    of the eigenvalues. One can try to vary ``k`` to make this
    better.
    2. How well conditioned the problem is. This can be changed by
    using proper preconditioning.
    """

    if method == 'ortho':
        return robust_lobpcg(A, k= k, B=B, X=X, niter=maxiter, iK=M, tol=tol, largest=largest, method='ortho',\
            verbosityLevel = verbosityLevel, ortho_iparams = ortho_iparams, ortho_fparams = ortho_fparams,\
            ortho_bparams= ortho_bparams, retLambdaHistory = retLambdaHistory, retResidualNormsHistory= retResidualNormsHistory)
    elif method == 'basic':
        return robust_lobpcg(A, k= k, B=B, X=X, niter=maxiter, iK=M, tol=tol, largest=largest, method='basic',\
                verbosityLevel = verbosityLevel, ortho_iparams = ortho_iparams, ortho_fparams = ortho_fparams,\
                ortho_bparams= ortho_bparams, retLambdaHistory = retLambdaHistory, retResidualNormsHistory= retResidualNormsHistory)
    elif method == 'blopex':
        X = cp.random.randn(A, k, dtype=A.dtype) if X is None else X
        return blopex_lobpcg(A, X, B=B, M=M, tol=tol, maxiter=maxiter, largest=largest, verbosityLevel=verbosityLevel,\
                retLambdaHistory=retLambdaHistory, retResidualNormsHistory=retResidualNormsHistory)

