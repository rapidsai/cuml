import cupy as cp
from scipy.sparse import csr_matrix as scipy_csr_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matirx
from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
from scipy.sparse.linalg import eigsh as scipy_eigsh
from cuml.manifold.lanczos import eig_lanczos
import numpy as np


X = [[4, 2], [2, 3]]
X = [[2, 2, 2], [2, 5, 5], [2, 5, 11]]
X = [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]
X = np.asarray(X, dtype=np.float32)


X = scipy_csr_matrix(X)
cupy_X = cupy_csr_matirx(X)

k=2
n=5
ncv = min(max(2 * k, k + 32), n - 1)
ncv = ncv-k

eigenvalues_raft, eigenvectors, eig_iters = eig_lanczos(X, k, 42, dtype=np.float32, maxiter=10000, tol=1e-15, conv_n_iters=4, conv_eps=0.001, restartiter=ncv, handle=None)

print("eigenvalues", eigenvalues_raft)

eigenvalues_cupy, eigenvectors_cupy = cupy_eigsh(cupy_X, k=k, which="SA", maxiter=10000, tol=0)

print("eigenvalues", eigenvalues_cupy)

eigenvalues_scipy, eigenvectors_scipy = scipy_eigsh(X, k=4, which="SA", maxiter=10000, tol=0)

print("eigenvalues", eigenvalues_scipy)




