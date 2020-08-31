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
import _cp_linalg_utils as _utils


__all__ = ['lobpcg']

def isNaN(num):
    return num!=num


def lobpcg(A,
           k=None,
           B=None,
           X=None,
           n=None,
           iK=None,
           niter=None,
           tol=None,
           largest=None,
           method=None,
           ortho_iparams=None,
           ortho_fparams=None,
           ortho_bparams=None,
           verbosityLevel = 0,
           retLambdaHistory=False,
           retResidualNormsHistory=False
           ):
    """
    This method implements the `method = "ortho"` and method = "basic"`
    LOBPCG algos.

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

    iK : cupy.ndarray, optional
            the input ndarray of size (n,n).
            When specified, it will be used as preconditioner.

    k : integer, optional (but needs to be specified if X isn't)
            the number of requested
            eigenpairs.

    niter : int, optional
            maximum number of iterations.
            When reached, the iteration process is hard-stopped and
            the current approximation of eigenpairs is returned.
            For infinite iteration but until convergence criteria
            is met, use `-1`.

    method : str, optional
            select LOBPCG method.
            `method="ortho"` - the Default LOBPCG method with
                orthogonal basis selection [StathopoulosEtal2002].
                A robust method.
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
    """

    # A must be square:
    assert A.shape[-2] == A.shape[-1], A.shape
    if B is not None:
        # A and B must have the same shapes:
        assert A.shape == B.shape, (A.shape, B.shape)

    #dtype = _utils.get_floating_dtype(A)
    dtype = cp.float32 if A.dtype not in (cp.float32, cp.float64) else A.dtype
    if tol is None:
        feps = 2.23e-16 if dtype == cp.float64 else 1.2e-07
        tol = feps ** 0.5

    m = A.shape[-1]
    k = (1 if X is None else X.shape[-1]) if k is None else k
    n = (k if n is None else n) if X is None else X.shape[-1]

    if (m < 3 * n):
        raise ValueError(
            'LPBPCG algorithm is not applicable when the number of A rows (={})'
            ' is smaller than 3 x the number of requested eigenpairs (={})'
            .format(m, n))

    method = 'ortho' if method is None else method

    iparams = {
        'm': m,
        'n': n,
        'k': k,
        'niter': 1000 if niter is None else niter,
    }

    fparams = {
        'tol': tol,
    }

    bparams = {
        'largest': True if largest is None else largest
    }

    if method == 'ortho':
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams['ortho_i_max'] = iparams.get('ortho_i_max', 3)
        iparams['ortho_j_max'] = iparams.get('ortho_j_max', 3)
        fparams['ortho_tol'] = fparams.get('ortho_tol', tol)
        fparams['ortho_tol_drop'] = fparams.get('ortho_tol_drop', tol)
        fparams['ortho_tol_replace'] = fparams.get('ortho_tol_replace', tol)
        bparams['ortho_use_drop'] = bparams.get('ortho_use_drop', False)

    if len(A.shape) > 2:
        N = int(cp.prod(cp.array(A.shape[:-2])))
        bA = A.reshape((N,) + A.shape[-2:])
        bB = B.reshape((N,) + A.shape[-2:]) if B is not None else None
        bX = X.reshape((N,) + X.shape[-2:]) if X is not None else None
        bE = cp.empty((N, k), dtype=dtype)
        bXret = cp.empty((N, m, k), dtype=dtype)
        bLambdaHistory = [None]*N
        bResidualNormsHistory = [None]*N

        for i in range(N):
            A_ = bA[i]
            B_ = bB[i] if bB is not None else None
            X_ = cp.random.randn(m, n, dtype=dtype) if bX is None else bX[i]
            assert len(X_.shape) == 2 and X_.shape == (m, n), (X_.shape, (m, n))
            iparams['batch_index'] = i
            worker = LOBPCG(A_, B_, X_, iK, iparams, fparams, bparams, method, verbosityLevel)
            worker.run()
            bE[i] = worker.E[:k]
            bXret[i] = worker.X[:, :k]
            bLambdaHistory[i] = worker.lambdaHistory
            bResidualNormsHistory[i] = worker.residualNormsHistory

        if retLambdaHistory:
            if retResidualNormsHistory:
                return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k)), bLambdaHistory, bResidualNormsHistory
            else:
                return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k)), bLambdaHistory
        else:
            if retResidualNormsHistory:
                return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k)), bResidualNormsHistory
            else:
                return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k))

    X = cp.random.randn(m, n, dtype=dtype) if X is None else X
    assert len(X.shape) == 2 and X.shape == (m, n), (X.shape, (m, n))

    worker = LOBPCG(A, B, X, iK, iparams, fparams, bparams, method,verbosityLevel)

    worker.run()

    if retLambdaHistory:
        if retResidualNormsHistory:
            return worker.E[:k], worker.X[:,:k], worker.lambdaHistory, worker.residualNormsHistory
        else:
            return worker.E[:k], worker.X[:,:k], worker.lambdaHistory
    else:
        if retResidualNormsHistory:
            return worker.E[:k], worker.X[:,:k], worker.residualNormsHistory
        else:
            return worker.E[:k], worker.X[:,:k]




class LOBPCG(object):
    """Worker class of LOBPCG methods.
    """

    def __init__(self,
                 A,
                 B,
                 X,
                 iK,
                 iparams,
                 fparams,
                 bparams,
                 method,
                 verbosityLevel
                 ):

        # constant parameters
        self.A = A
        self.B = B
        self.iK = iK
        self.iparams = iparams
        self.fparams = fparams
        self.bparams = bparams
        self.method = method
        self.verbosityLevel = verbosityLevel
        m = iparams['m']
        n = iparams['n']

        # variable parameters
        self.X = X
        self.E = cp.zeros((n, ), dtype=X.dtype)
        self.R = cp.zeros((m, n), dtype=X.dtype)
        self.S = cp.zeros((m, 3 * n), dtype=X.dtype)
        self.tvars = {}               # type: Dict[str, ndarray]
        self.ivars = {'istep': 0}     # type: Dict[str, int]
        self.fvars = {'_': 0.0}       # type: Dict[str, float]
        self.bvars = {'_': False}     # type: Dict[str, bool]
        self.lambdaHistory = []
        self.residualNormsHistory = []

    def __str__(self):
        lines = ['LOPBCG:']
        lines += ['  iparams={}'.format(self.iparams)]
        lines += ['  fparams={}'.format(self.fparams)]
        lines += ['  bparams={}'.format(self.bparams)]
        lines += ['  ivars={}'.format(self.ivars)]
        lines += ['  fvars={}'.format(self.fvars)]
        lines += ['  bvars={}'.format(self.bvars)]
        lines += ['  tvars={}'.format(self.tvars)]
        lines += ['  A={}'.format(self.A)]
        lines += ['  B={}'.format(self.B)]
        lines += ['  iK={}'.format(self.iK)]
        lines += ['  X={}'.format(self.X)]
        lines += ['  E={}'.format(self.E)]
        r = ''
        for line in lines:
            r += line + '\n'
        return r

    def update(self):
        """Set and update iteration variables.
        """
        if self.ivars['istep'] == 0:
            X_norm = float(cp.linalg.norm(self.X))
            iX_norm = X_norm ** -1
            A_norm = float(cp.linalg.norm(_utils.matmul(self.A, self.X))) * iX_norm
            B_norm = float(cp.linalg.norm(_utils.matmul(self.B, self.X))) * iX_norm
            self.fvars['X_norm'] = X_norm
            self.fvars['A_norm'] = A_norm
            self.fvars['B_norm'] = B_norm
            self.ivars['iterations_left'] = self.iparams['niter']
            self.ivars['converged_count'] = 0
            self.ivars['converged_end'] = 0

        if self.method == 'ortho':
            self._update_ortho()
        else:
            self._update_basic()

        self.ivars['iterations_left'] = self.ivars['iterations_left'] - 1
        self.ivars['istep'] = self.ivars['istep'] + 1
        self.lambdaHistory.append(self.E)

    def update_residual(self):
        """Update residual R from A, B, X, E.
            Calculate the residual norms
        """
        mm = _utils.matmul
        self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E
        aux = cp.sum(self.R.conj() * self.R, 0)
        residualNorm = cp.sqrt(aux)
        self.residualNormsHistory.append(residualNorm)

    def update_converged_count(self):
        """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].
        Users may redefine this method for custom convergence criteria.
        """
        # (...) -> int
        prev_count = self.ivars['converged_count']
        tol = self.fparams['tol']
        A_norm = self.fvars['A_norm']
        B_norm = self.fvars['B_norm']
        E, X, R = self.E, self.X, self.R
        rerr = cp.linalg.norm(R, 2, (0, )) * (cp.linalg.norm(X, 2, (0, )) * (A_norm + E[:X.shape[-1]] * B_norm)) ** -1
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                # ignore convergence of following pairs to ensure
                # strict ordering of eigenpairs
                break
            count += 1
        assert count >= prev_count, 'the number of converged eigenpairs ' \
            '(was {}, got {}) cannot decrease'.format(prev_count, count)
        self.ivars['converged_count'] = count
        self.tvars['rerr'] = rerr
        return count

    def stop_iteration(self):
        """Return True to stop iterations.
        """
        return (self.bvars.get('force_stop', False)
                or self.ivars['iterations_left'] == 0
                or self.ivars['converged_count'] >= self.iparams['k'])

    def run(self):
        """Run LOBPCG iterations.
        Use this method as a template for implementing LOBPCG
        """
        self.update()

        if(self.verbosityLevel > 0):
            print(self.__str__())

        while not self.stop_iteration():

            self.update()
            if(self.verbosityLevel > 0):
                print(self.__str__())



    def _update_basic(self):
        """
        Update or initialize iteration variables when `method == "basic"`.
        """
        mm = cp.matmul
        ns = self.ivars['converged_end']
        nc = self.ivars['converged_count']
        n = self.iparams['n']
        largest = self.bparams['largest']

        if self.ivars['istep'] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            E, Z = _utils.symeig(M, largest)
            self.X[:] = mm(self.X, mm(Ri, Z))
            self.E[:] = E
            np = 0
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X

            W = _utils.matmul(self.iK, self.R)
            self.ivars['converged_end'] = ns = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W
        else:
            S_ = self.S[:, nc:ns]
            Ri = self._get_rayleigh_ritz_transform(S_)
            M = _utils.qform(_utils.qform(self.A, S_), Ri)
            E_, Z = _utils.symeig(M, largest)
            self.X[:, nc:] = mm(S_, mm(Ri, Z[:, :n - nc]))
            self.E[nc:] = E_[:n - nc]
            P = mm(S_, mm(Ri, Z[:, n:2 * n - nc]))
            np = P.shape[-1]

            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            self.S[:, n:n + np] = P
            W = _utils.matmul(self.iK, self.R[:, nc:])

            self.ivars['converged_end'] = ns = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W

    def _update_ortho(self):
        """
        Update or initialize iteration variables when `method == "ortho"`.
        """
        mm = cp.matmul
        ns = self.ivars['converged_end']
        nc = self.ivars['converged_count']
        n = self.iparams['n']
        largest = self.bparams['largest']

        if self.ivars['istep'] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            E, Z = _utils.symeig(M, largest)
            self.X = mm(self.X, mm(Ri, Z))
            self.update_residual()
            np = 0
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            W = self._get_ortho(self.R, self.X)
            ns = self.ivars['converged_end'] = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W

        else:
            S_ = self.S[:, nc:ns]
            # Rayleigh-Ritz procedure
            E_, Z = _utils.symeig(_utils.qform(self.A, S_), largest)

            # Update E, X, P
            self.X[:, nc:] = mm(S_, Z[:, :n - nc])
            self.E[nc:] = E_[:n - nc]
            P = mm(S_, mm(Z[:, n - nc:], _utils.basis(_utils.transpose(Z[:n - nc, n - nc:]))))
            np = P.shape[-1]

            # check convergence
            self.update_residual()
            nc = self.update_converged_count()

            # update S
            self.S[:, :n] = self.X
            self.S[:, n:n + np] = P
            W = self._get_ortho(self.R[:, nc:], self.S[:, :n + np])
            ns = self.ivars['converged_end'] = n + np + W.shape[-1]
            self.S[:, n + np:ns] = W

    def _get_rayleigh_ritz_transform(self, S):
        """Return a transformation matrix that is used in Rayleigh-Ritz
        procedure for reducing a general eigenvalue problem :math:`(S^TAS)
        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
        S^TAS Ri) Z = Z E` where `C = Ri Z`.
        .. note:: In the original Rayleight-Ritz procedure in
          [DuerschEtal2018], the problem is formulated as follows::
            SAS = S^T A S
            SBS = S^T B S
            D = (<diagonal matrix of SBS>) ** -1/2
            R^T R = Cholesky(D SBS D)
            Ri = D R^-1
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z
          To reduce the number of matrix products (denoted by empty
          space between matrices), here we introduce element-wise
          products (denoted by symbol `*`) so that the Rayleight-Ritz
          procedure becomes::
            SAS = S^T A S
            SBS = S^T B S
            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector
            dd = d d^T                         # this is 2-d matrix
            R^T R = Cholesky(dd * SBS)
            Ri = R^-1 * d                      # broadcasting
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z
          where `dd` is 2-d matrix that replaces matrix products `D M
          D` with one element-wise product `M * dd`; and `d` replaces
          matrix product `D M` with element-wise product `M *
          d`. Also, creating the diagonal matrix `D` is avoided.
        Parameters
        ------------
        S (ndarray): the matrix basis for the search subspace, size is
                    :math:`(m, n)`.
        Returns
        -------
        Ri (ndarray): upper-triangular transformation matrix of size
                     :math:`(n, n)`.
        """
        B = self.B
        mm = cp.matmul
        SBS = _utils.qform(B, S)
        d_row = SBS.diagonal(0, -2, -1) ** -0.5
        d_col = d_row.reshape(d_row.shape[0], 1)
        ch_inp= (SBS * d_row) * d_col
        R = cp.linalg.cholesky((SBS * d_row) * d_col).transpose()#cupy linalg cholesky returns the lower triangular matrix. we transpose to get the upper one.
        Rinv = cp.linalg.inv(R)
        return Rinv * d_col

    def _get_svqb(self,
                  U,     # ndarray
                  drop,  # bool
                  tau    # float
                  ):
        # type: (ndarray, bool, float) -> ndarray
        """Return B-orthonormal U.
        .. note:: When `drop` is `False` then `svqb` is based on the
                  Algorithm 4 from [DuerschPhD2015] that is a slight
                  modification of the corresponding algorithm
                  introduced in [StathopolousWu2002].
        Parameters
        ------------
          U (ndarray) : initial approximation, size is (m, n)
          drop (bool) : when True, drop columns that
                     contribution to the `span([U])` is small.
          tau (float) : positive tolerance
        Returns
        -------
          U (ndarray) : B-orthonormal columns (:math:`U^T B U = I`), size
                       is (m, n1), where `n1 = n` if `drop` is `False,
                       otherwise `n1 <= n`.
        """
        if U.size == 0:
            return U
        UBU = _utils.qform(self.B, U)
        d = UBU.diagonal(0, -2, -1)

        # Detect and drop exact zero columns from U. While the test
        # `abs(d) == 0` is unlikely to be True for random data, it is
        # possible to construct input data to lobpcg where it will be
        # True leading to a failure (notice the `d ** -0.5` operation
        # in the original algorithm). To prevent the failure, we drop
        # the exact zero columns here and then continue with the
        # original algorithm below.
        nz = cp.where(cp.abs(d) != 0.0)
        assert len(nz) == 1, "NonZero Values of d:{}".format(nz)
        if len(nz[0]) < len(d):
            U = U[:, nz[0]]
            if len(U) == 0:
                return U
            UBU = _utils.qform(self.B, U)
            d = UBU.diagonal(0, -2, -1)
            nz = cp.where(cp.abs(d) != 0.0)
            assert len(nz[0]) == len(d)

        # The original algorithm 4 from [DuerschPhD2015].
        d_col = (d ** -0.5).reshape(d.shape[0], 1)
        DUBUD = (UBU * d_col) * _utils.transpose(d_col)
        E, Z = _utils.symeig(DUBUD)
        t = tau * cp.abs(E).max()
        if drop:
            keep = cp.where(E > t)
            assert len(keep) == 1, keep
            E = E[keep[0]]
            Z = Z[:, keep[0]]
            d_col = d_col[keep[0]]
        else:
            E[(cp.where(E < t))[0]] = t

        return cp.matmul(U * _utils.transpose(d_col), Z * E ** -0.5)

    def _get_ortho(self, U, V):
        """Return B-orthonormal U with columns are B-orthogonal to V.
        .. note:: When `bparams["ortho_use_drop"] == False` then
                  `_get_ortho` is based on the Algorithm 3 from
                  [DuerschPhD2015] that is a slight modification of
                  the corresponding algorithm introduced in
                  [StathopolousWu2002]. Otherwise, the method
                  implements Algorithm 6 from [DuerschPhD2015]
        .. note:: If all U columns are B-collinear to V then the
                  returned ndarray U will be empty.
        Parameters
        ----------

          U (ndarray) : initial approximation, size is (m, n)
          V (ndarray) : B-orthogonal external basis, size is (m, k)

        Returns
        -------
          U (ndarray) : B-orthonormal columns (:math:`U^T B U = I`)
                       such that :math:`V^T B U=0`, size is (m, n1),
                       where `n1 = n` if `drop` is `False, otherwise
                       `n1 <= n`.
        """
        mm = cp.matmul
        mm_B = _utils.matmul
        m = self.iparams['m']
        tau_ortho = self.fparams['ortho_tol']
        tau_drop = self.fparams['ortho_tol_drop']
        tau_replace = self.fparams['ortho_tol_replace']
        i_max = self.iparams['ortho_i_max']
        j_max = self.iparams['ortho_j_max']
        # when use_drop==True, enable dropping U columns that have
        # small contribution to the `span([U, V])`.
        use_drop = self.bparams['ortho_use_drop']

        # clean up variables from the previous call
        for vkey in list(self.fvars.keys()):
            if vkey.startswith('ortho_') and vkey.endswith('_rerr'):
                self.fvars.pop(vkey)
        self.ivars.pop('ortho_i', 0)
        self.ivars.pop('ortho_j', 0)

        BV_norm = cp.linalg.norm(mm_B(self.B, V))
        BU = mm_B(self.B, U)
        VBU = mm(_utils.transpose(V), BU)
        i = j = 0
        stats = ''
        for i in range(i_max):
            U = U - mm(V, VBU)
            drop = False
            tau_svqb = tau_drop
            for j in range(j_max):
                if use_drop:
                    U = self._get_svqb(U, drop, tau_svqb)
                    drop = True
                    tau_svqb = tau_replace
                else:
                    U = self._get_svqb(U, False, tau_replace)
                if U.size == 0:
                    # all initial U columns are B-collinear to V
                    self.ivars['ortho_i'] = i
                    self.ivars['ortho_j'] = j
                    return U
                BU = mm_B(self.B, U)
                UBU = mm(_utils.transpose(U), BU)
                U_norm = cp.linalg.norm(U)
                BU_norm = cp.linalg.norm(BU)
                R = UBU - cp.eye(UBU.shape[-1],
                                    dtype=UBU.dtype)
                R_norm = cp.linalg.norm(R)
                rerr = float(R_norm) * float(BU_norm * U_norm) ** -1
                vkey = 'ortho_UBUmI_rerr[{}, {}]'.format(i, j)
                self.fvars[vkey] = rerr
                if rerr < tau_ortho:
                    break
            VBU = mm(_utils.transpose(V), BU)
            VBU_norm = cp.linalg.norm(VBU)
            U_norm = cp.linalg.norm(U)
            rerr = float(VBU_norm) * float(BV_norm * U_norm) ** -1
            vkey = 'ortho_VBU_rerr[{}]'.format(i)
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
            if m < U.shape[-1] + V.shape[-1]:
                B = self.B
                assert B is not None
                raise ValueError(
                    'Overdetermined shape of U:'
                    ' #B-cols(={}) >= #U-cols(={}) + #V-cols(={}) must hold'
                    .format(B.shape[-1], U.shape[-1], V.shape[-1]))
        self.ivars['ortho_i'] = i
        self.ivars['ortho_j'] = j
        return U
