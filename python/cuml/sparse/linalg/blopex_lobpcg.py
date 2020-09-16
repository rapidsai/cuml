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

"""
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).
References
----------
.. [1] A. V. Knyazev (2001),
       Toward the Optimal Preconditioned Eigensolver: Locally Optimal
       Block Preconditioned Conjugate Gradient Method.
       SIAM Journal on Scientific Computing 23, no. 2,
       pp. 517-541. http://dx.doi.org/10.1137/S1064827500366124
.. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),
       Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)
       in hypre and PETSc.  https://arxiv.org/abs/0705.2626
.. [3] A. V. Knyazev's C and MATLAB implementations:
       https://github.com/lobpcg/blopex
"""

import cupy as cp
from cupy.linalg import (solve, cholesky, inv, eigh)
from cupyx.scipy.sparse import (spmatrix, issparse)
# import warnings
from warnings import warn

__all__ = ['lobpcg']


def _report_nonhermitian(M, name):
    """
    Report if `M` is not a hermitian matrix given its type.
    """
    from cupy.linalg import norm

    md = M - M.T.conj()

    nmd = norm(md, 1)
    tol = 10 * cp.finfo(M.dtype).eps
    tol = max(tol, tol * norm(M, 1))
    if nmd > tol:
        print('matrix %s of the type %s is not sufficiently Hermitian:' %
              (name, M.dtype))
        print('condition: %.e < %e' % (nmd, tol))


def _as2d(ar):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
    if ar.ndim == 2:
        return ar
    else:  # Assume 1!
        aux = cp.array(ar, copy=False)
        aux.shape = (ar.shape[0], 1)
        return aux


def _applyConstraints(blockVectorV, YBY, blockVectorBY, blockVectorY):
    """Changes blockVectorV in place."""
    YBV = cp.dot(blockVectorBY.T.conj(), blockVectorV)
    tmp = solve(YBY, YBV)
    blockVectorV -= cp.dot(blockVectorY, tmp)


def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, retInvR=False):
    """B-orthonormalize the given block vector using Cholesky."""

    normalization = blockVectorV.max(axis=0) + cp.finfo(blockVectorV.dtype).eps
    blockVectorV = blockVectorV / normalization
    if blockVectorBV is None:
        if B is not None:
            blockVectorBV = cp.matmul(B, blockVectorV)
        else:
            blockVectorBV = blockVectorV  # Shared data!!!
    else:
        blockVectorBV = blockVectorBV / normalization
    VBV = cp.matmul(blockVectorV.T.conj(), blockVectorBV)
    try:
        # VBV is a Cholesky factor from now on...
        # cupy cholesky module returns lower triangular matrix!!!
        VBV = cp.transpose(cholesky(VBV))
        VBV = inv(VBV)
        blockVectorV = cp.matmul(blockVectorV, VBV)
        if B is not None:
            blockVectorBV = cp.matmul(blockVectorBV, VBV)
        else:
            blockVectorBV = None
    except Exception as e:
        # raise ValueError('Cholesky has failed')
        print("{} occured".format(e))
        blockVectorV = None
        blockVectorBV = None
        VBV = None

    if retInvR:
        return blockVectorV, blockVectorBV, VBV, normalization
    else:
        return blockVectorV, blockVectorBV


def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = cp.argsort(_lambda)
    if largest:
        ii = ii[:-num - 1:-1]
    else:
        ii = ii[:num]
    return ii


def _genEigh(A, B):
    """
    Helper function for converting a generalized eigenvalue problem
    AX = lambdaBX to standard using cholesky.
    This is because cupy does not have a functional api to solve
    generalized eigenvalue problem.
    Factorizing B = R^TR. Let F = (R^T)^-1 A R^-1
    Equivalent Standard form: Fy = lambda(y), where our required
    eigvec x = R^-1 y
    """
    # we transpose lower triangular matrix to get upper triangular matrix
    R = cp.transpose(cholesky(B))
    RTi = inv(cp.transpose(R))
    Ri = inv(R)
    F = cp.matmul(RTi, cp.matmul(A, Ri))
    vals, vecs = eigh(F)
    eigVec = cp.matmul(Ri, vecs)

    return vals, eigVec


def _bmat(list_obj):
    """
    Helper function to create a block matrix in cupy from a list
    of smaller 2D matrices using iterated vertical and horizontal stacking
    """

    arr_rows = cp.array([])
    for row in list_obj:  # row & list_obj are a list of cupy arrays
        arr_cols = cp.array([])
        for col in row:  # col is a cupy ndarray
            if col.ndim == 0:
                col = cp.array([col])
            if(arr_cols.size == 0):
                arr_cols = col
                continue

            arr_cols = cp.hstack((arr_cols, col))
        if(arr_rows.size == 0):
            arr_rows = arr_cols
            continue
        arr_rows = cp.vstack((arr_rows, arr_cols))

    return arr_rows


def _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel):
    if verbosityLevel > 0:
        _report_nonhermitian(gramA, 'gramA')
        _report_nonhermitian(gramB, 'gramB')


def lobpcg(A,
           X,
           B=None,
           M=None,
           Y=None,
           tol=None,
           maxiter=None,
           largest=True,
           verbosityLevel=0,
           retLambdaHistory=False,
           retResidualNormsHistory=False):
    """Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)
    LOBPCG is a preconditioned eigensolver for large symmetric positive
    definite (SPD) generalized eigenproblems.
    Parameters
    ----------
    A : {cupy.ndarray}
        The symmetric linear operator of the problem, Is treated as a 2D matrix
        Often called the "stiffness matrix".
    X : cupy.ndarray, float32 or float64
        Initial approximation to the ``k`` eigenvectors (non-sparse). If `A`
        has ``shape=(n,n)`` then `X` should have shape ``shape=(n,k)``.
    B : {cupy ndarray}, optional
        The right hand side operator in a generalized eigenproblem.
        By default, ``B = Identity``.  Often called the "mass matrix".
    M : {cupy ndarray}, optional
        Preconditioner to `A`; by default ``M = Identity``.
        `M` should approximate the inverse of `A`.
    Y : ndarray, float32 or float64, optional
        n-by-sizeY matrix of constraints (non-sparse), sizeY < n
        The iterations will be performed in the B-orthogonal complement
        of the column-space of Y. Y must be full rank.
    tol : scalar, optional
        Solver tolerance (stopping criterion).
        The default is ``tol=n*sqrt(eps)``.
    maxiter : int, optional
        Maximum number of iterations.  The default is ``maxiter = 20``.
    largest : bool, optional
        When True, solve for the largest eigenvalues, otherwise the smallest.
    verbosityLevel : int, optional
        Controls solver output.  The default is ``verbosityLevel=0``.
    retLambdaHistory : bool, optional
        Whether to return eigenvalue history.  Default is False.
    retResidualNormsHistory : bool, optional
        Whether to return history of residual norms.  Default is False.
    Returns
    -------
    w : ndarray
        Array of ``k`` eigenvalues
    v : ndarray
        An array of ``k`` eigenvectors.  `v` has the same shape as `X`.
    lambdas : list of ndarray, optional
        The eigenvalue history, if `retLambdaHistory` is True.
    rnorms : list of ndarray, optional
        The history of residual norms, if `retResidualNormsHistory` is True.

    """

    blockVectorX = X
    blockVectorY = Y
    residualTolerance = tol
    if maxiter is None:
        maxiter = 20

    if blockVectorY is not None:
        sizeY = blockVectorY.shape[1]
    else:
        sizeY = 0

    # Block size.
    if len(blockVectorX.shape) != 2:
        raise ValueError('expected rank-2 array for argument X')

    n, sizeX = blockVectorX.shape

    if verbosityLevel:
        aux = "Solving "
        if B is None:
            aux += "standard"
        else:
            aux += "generalized"
        aux += " eigenvalue problem with"
        if M is None:
            aux += "out"
        aux += " preconditioning\n\n"
        aux += "matrix size %d\n" % n
        aux += "block size %d\n\n" % sizeX
        if blockVectorY is None:
            aux += "No constraints\n\n"
        else:
            if sizeY > 1:
                aux += "%d constraints\n\n" % sizeY
            else:
                aux += "%d constraint\n\n" % sizeY
        print(aux)

    if (n - sizeY) < (5 * sizeX):
        warn('The problem size is small compared to the block size!'
             ' Using the standard eigen solver instead of LOBPCG.')

        sizeX = min(sizeX, n)

        if blockVectorY is not None:
            raise NotImplementedError('The dense eigensolver '
                                      'does not support constraints.')

        A_dense = spmatrix.toarray(A) if issparse(A) else A
        B_dense = spmatrix.toarray(B) if issparse(B) else B

        vals, vecs = _genEigh(A_dense, B_dense)
        if largest:
            vals = vals[::-1]
            vecs = vecs[:, ::-1]

        return vals[:sizeX], vecs[:, :sizeX]

    if (residualTolerance is None) or (residualTolerance <= 0.0):
        residualTolerance = cp.sqrt(1e-15) * n

    # Apply constraints to X.
    if blockVectorY is not None:

        if B is not None:
            blockVectorBY = cp.matmul(B, blockVectorY)
        else:
            blockVectorBY = blockVectorY

        # gramYBY is a dense array.
        gramYBY = cp.dot(blockVectorY.T.conj(), blockVectorBY)

        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)

    ##
    # B-orthonormalize X.
    blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)

    ##
    # Compute the initial Ritz vectors: solve the eigenproblem.
    blockVectorAX = cp.matmul(A, blockVectorX)
    gramXAX = cp.dot(blockVectorX.T.conj(), blockVectorAX)

    _lambda, eigBlockVector = eigh(gramXAX)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]

    eigBlockVector = cp.asarray(eigBlockVector[:, ii])
    blockVectorX = cp.dot(blockVectorX, eigBlockVector)
    blockVectorAX = cp.dot(blockVectorAX, eigBlockVector)
    if B is not None:
        blockVectorBX = cp.dot(blockVectorBX, eigBlockVector)

    ##
    # Active index set.
    activeMask = cp.ones((sizeX, ), dtype=bool)

    lambdaHistory = [_lambda]
    residualNormsHistory = []

    previousBlockSize = sizeX
    ident = cp.eye(sizeX, dtype=A.dtype)
    ident0 = cp.eye(sizeX, dtype=A.dtype)

    ##
    # Main iteration loop.

    blockVectorP = None  # set during iteration
    blockVectorAP = None
    blockVectorBP = None

    iterationNumber = -1
    restart = True
    explicitGramFlag = False
    while iterationNumber < maxiter:
        iterationNumber += 1
        if verbosityLevel > 0:
            print('iteration %d' % iterationNumber)

        if B is not None:
            aux = blockVectorBX * _lambda[cp.newaxis, :]
        else:
            aux = blockVectorX * _lambda[cp.newaxis, :]

        blockVectorR = blockVectorAX - aux

        aux = cp.sum(blockVectorR.conj() * blockVectorR, 0)
        residualNorms = cp.sqrt(aux)

        residualNormsHistory.append(residualNorms)

        ii = cp.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii
        if verbosityLevel > 2:
            print('the mask depending on var largest:', activeMask)

        currentBlockSize = activeMask.sum()
        currentBlockSize = int(currentBlockSize)

        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = cp.eye(currentBlockSize, dtype=A.dtype)

        if currentBlockSize == 0:
            break

        if verbosityLevel > 0:
            print('current block size:', currentBlockSize)
            print('eigenvalue:', _lambda)
            print('residual norms:', residualNorms)
        if verbosityLevel > 10:
            print('Eigen Block Vector:', eigBlockVector)

        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])

        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])

        if M is not None:
            # Apply preconditioner T to the active residuals.
            activeBlockVectorR = cp.matmul(M, activeBlockVectorR)

        ##
        # Apply constraints to the preconditioned residuals.
        if blockVectorY is not None:
            _applyConstraints(activeBlockVectorR, gramYBY, blockVectorBY,
                              blockVectorY)

        ##
        # B-orthogonalize the preconditioned residuals to X.
        if B is not None:
            activeBlockVectorR = activeBlockVectorR - cp.matmul(
                blockVectorX,
                cp.matmul(blockVectorBX.T.conj(), activeBlockVectorR))
        else:
            activeBlockVectorR = activeBlockVectorR - cp.matmul(
                blockVectorX,
                cp.matmul(blockVectorX.T.conj(), activeBlockVectorR))

        ##
        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux

        activeBlockVectorAR = cp.matmul(A, activeBlockVectorR)

        if iterationNumber > 0:
            if B is not None:
                aux = _b_orthonormalize(B,
                                        activeBlockVectorP,
                                        activeBlockVectorBP,
                                        retInvR=True)
                activeBlockVectorP, activeBlockVectorBP, invR, normal = aux
            else:
                aux = _b_orthonormalize(B, activeBlockVectorP, retInvR=True)
                activeBlockVectorP, _, invR, normal = aux
            # Function _b_orthonormalize returns None if Cholesky fails
            if activeBlockVectorP is not None:
                activeBlockVectorAP = activeBlockVectorAP / normal
                activeBlockVectorAP = cp.dot(activeBlockVectorAP, invR)
                restart = False
            else:
                restart = True

        ##
        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:

        if activeBlockVectorAR.dtype == 'float32':
            myeps = 1
        elif activeBlockVectorR.dtype == 'float32':
            myeps = 1e-4
        else:
            myeps = 1e-8

        if residualNorms.max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assingments to simplify the code
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP

        # Common submatrices:
        gramXAR = cp.dot(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = cp.dot(activeBlockVectorR.T.conj(), activeBlockVectorAR)

        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = cp.dot(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = cp.dot(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = cp.dot(activeBlockVectorR.T.conj(), activeBlockVectorBR)
            gramXBR = cp.dot(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = cp.diag(_lambda)
            gramXBX = ident0
            gramRBR = ident
            assert isinstance(currentBlockSize, (int))
            gramXBR = cp.zeros((sizeX, currentBlockSize), dtype=A.dtype)

        if not restart:
            gramXAP = cp.dot(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = cp.dot(activeBlockVectorR.T.conj(), activeBlockVectorAP)
            gramPAP = cp.dot(activeBlockVectorP.T.conj(), activeBlockVectorAP)
            gramXBP = cp.dot(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = cp.dot(activeBlockVectorR.T.conj(), activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = cp.dot(activeBlockVectorP.T.conj(),
                                 activeBlockVectorBP)
            else:
                gramPBP = ident

            gramA = _bmat([[gramXAX, gramXAR, gramXAP],
                           [gramXAR.T.conj(), gramRAR, gramRAP],
                           [gramXAP.T.conj(),
                            gramRAP.T.conj(), gramPAP]])
            gramB = _bmat([[gramXBX, gramXBR, gramXBP],
                           [gramXBR.T.conj(), gramRBR, gramRBP],
                           [gramXBP.T.conj(),
                            gramRBP.T.conj(), gramPBP]])

            _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel)

            try:
                _lambda, eigBlockVector = _genEigh(gramA,
                                                   gramB)
            except Exception as e:
                # try again after dropping the direction vectors P from RR
                print('{} occured'.format(e))
                restart = True

        if restart:
            gramA = _bmat([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = _bmat([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])

            _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel)

            try:
                _lambda, eigBlockVector = _genEigh(gramA,
                                                   gramB)

            except Exception as e:
                raise ValueError('''eigh has failed in lobpcg iterations with
                                 with exception {}\n'''.format(e))

        ii = _get_indx(_lambda, sizeX, largest)
        if verbosityLevel > 10:
            print('indices', ii)
            print('lambda', _lambda)

        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]

        lambdaHistory.append(_lambda)

        if verbosityLevel > 10:
            print('order-updated lambda:', _lambda)

        #         # Normalize eigenvectors!
        #         aux = cp.sum( eigBlockVector.conj() * eigBlockVector, 0 )
        #         eigVecNorms = cp.sqrt( aux )
        #         eigBlockVector = eigBlockVector / eigVecNorms[cp.newaxis, :]
        #         eigBlockVector, aux = _b_orthonormalize( B, eigBlockVector )

        if verbosityLevel > 10:
            print('eigenblockvector so far:', eigBlockVector)

        # Compute Ritz vectors.
        if B is not None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX +
                                                 currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = cp.dot(activeBlockVectorR, eigBlockVectorR)
                pp += cp.dot(activeBlockVectorP, eigBlockVectorP)

                app = cp.dot(activeBlockVectorAR, eigBlockVectorR)
                app += cp.dot(activeBlockVectorAP, eigBlockVectorP)

                bpp = cp.dot(activeBlockVectorBR, eigBlockVectorR)
                bpp += cp.dot(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = cp.dot(activeBlockVectorR, eigBlockVectorR)
                app = cp.dot(activeBlockVectorAR, eigBlockVectorR)
                bpp = cp.dot(activeBlockVectorBR, eigBlockVectorR)

            if verbosityLevel > 10:
                print('other derived vectors from res, A, B:')
                print(pp)
                print(app)
                print(bpp)

            blockVectorX = cp.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = cp.dot(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = cp.dot(blockVectorBX, eigBlockVectorX) + bpp

            blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp

        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX +
                                                 currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = cp.dot(activeBlockVectorR, eigBlockVectorR)
                pp += cp.dot(activeBlockVectorP, eigBlockVectorP)

                app = cp.dot(activeBlockVectorAR, eigBlockVectorR)
                app += cp.dot(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = cp.dot(activeBlockVectorR, eigBlockVectorR)
                app = cp.dot(activeBlockVectorAR, eigBlockVectorR)

            if verbosityLevel > 10:
                print('other derived vectors from res, A, B:')
                print(pp)
                print(app)

            blockVectorX = cp.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = cp.dot(blockVectorAX, eigBlockVectorX) + app

            blockVectorP, blockVectorAP = pp, app

    if B is not None:
        aux = blockVectorBX * _lambda[cp.newaxis, :]

    else:
        aux = blockVectorX * _lambda[cp.newaxis, :]

    blockVectorR = blockVectorAX - aux

    aux = cp.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = cp.sqrt(aux)

    # Future work: Need to add Postprocessing here:
    # Making sure eigenvectors "exactly" satisfy the blockVectorY constrains?
    # Making sure eigenvecotrs are "exactly" othonormalized by final "exact" RR
    # Computing the actual true residuals

    if verbosityLevel > 0:
        print('final eigenvalue:', _lambda)
        print('final residual norms:', residualNorms)

    if retLambdaHistory:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else:
            return _lambda, blockVectorX, lambdaHistory
    else:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, residualNormsHistory
        else:
            print("val: {}\n vec:{}\n".format(_lambda, blockVectorX))
            return _lambda, blockVectorX
