import numpy as np
import struct
from scipy.sparse import csr_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matirx
from cuml.manifold.lanczos import eig_lanczos
from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
from scipy.sparse.linalg import eigsh as scipy_eigsh
import cupy as cp
import time
from scipy.sparse.linalg import ArpackError

# np.set_printoptions(linewidth=np.inf, precision=7, suppress=True, threshold=np.inf, floatmode='fixed')

def load_vectors(filename):
    with open(filename, 'rb') as f:
        # Read the size of each vector
        size_rows = struct.unpack('Q', f.read(8))[0]
        size_cols = struct.unpack('Q', f.read(8))[0]
        size_vals = struct.unpack('Q', f.read(8))[0]

        # Read the vectors
        rows = np.fromfile(f, dtype=np.int32, count=size_rows)
        cols = np.fromfile(f, dtype=np.int32, count=size_cols)
        vals = np.fromfile(f, dtype=np.float32, count=size_vals)

    return rows, cols, vals

# Load the vectors
rows, cols, vals = load_vectors('/home/coder/raft/cpp/build/latest/gtests/sparse.bin')
n = rows.shape[0] - 1

print(rows.shape)
print(cols.shape)
print(vals.shape)
print("Rows:", rows)
print("Cols:", cols)
print("Vals:", vals)

X = csr_matrix((vals, cols, rows), shape=(n, n), dtype=np.float32)
X_gpu = cupy_csr_matirx(X, dtype=cp.float32)
print(X.shape, X.nnz)

k=50
n_components=k
ncv = min(max(2 * k, k + 32), n - 1)
ncv = ncv-k
print("ncv print", ncv+k)

start = time.time()
eigenvalues_raft, eigenvectors, eig_iters = eig_lanczos(X_gpu, k, 42, dtype=np.float32, maxiter=10000, tol=1e-9, conv_n_iters=4, conv_eps=0.001, restartiter=ncv, handle=None)
print("Raft Time (s)", time.time() - start)
print("eigenvalues", np.array2string(eigenvalues_raft, separator=', '), eig_iters)
print(eigenvectors[:, 0][:10])


start = time.time()
eigenvalues_cupy, eigenvectors = cupy_eigsh(X_gpu, k=k, which="SA", maxiter=10000, tol=1e-9)
print("Cupy eigsh Sklearn Time (s)", time.time() - start)
print("eigenvalues", np.array2string(eigenvalues_cupy, separator=', '))
print(eigenvectors[:, 0][:10])


if (X is not None):
    try:
        start = time.time()
        eigenvalues_sklearn, eigenvectors = scipy_eigsh((X), k=k, which="SA", maxiter=10000, tol=0)
        print("Eigsh Sklearn Time (s)", time.time() - start)
        print("eigenvalues", np.array2string(eigenvalues_sklearn, separator=', '))
        print(eigenvectors[:, 0][:10])
    except ArpackError as e:
        print(e)

# if (X is not None):
#     try:
#         import scipy.linalg
#         start = time.time()
#         eigenvalues_sklearn, eigenvectors = scipy.linalg.eigh((X.toarray()))
#         print("Eigsh Sklearn Time (s)", time.time() - start)
#         print("eigenvalues", np.array2string(eigenvalues_sklearn, separator=', '))
#         print(eigenvectors[:, 0][:10])
#     except ArpackError as e:
#         print(e)

eigenvalues_raft = np.asarray(eigenvalues_raft)
eigenvalues_cupy = cp.asnumpy(eigenvalues_cupy)
if X is not None:
    eigenvalues_sklearn = np.asarray(eigenvalues_sklearn)

def compare_arr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Calculate the absolute difference
    diff = np.abs(x - y)
    # print(diff)
    # Find the maximum difference and its index
    max_diff = np.max(diff)
    max_diff_index = np.argmax(diff)
    
    # Print the results
    print("Maximum difference:", max_diff)
    print("Index of maximum difference:", max_diff_index)

are_close = np.allclose(eigenvalues_raft, eigenvalues_cupy, atol=0, rtol=1e-5)
print("Arrays are close:", are_close)
if X is not None:
    are_close = np.allclose(eigenvalues_raft, eigenvalues_sklearn, atol=0, rtol=1e-5)
    print("Arrays are close:", are_close)
    array1 = eigenvalues_raft
    array2 = eigenvalues_sklearn
    # Create a boolean array showing which elements are close
    close_mask = np.isclose(array1, array2, atol=0, rtol=1e-4)

    # Find the indices of elements that are not close
    not_close_indices = ~close_mask

    # Extract the elements from both arrays where they are not close
    elements_not_close_array1 = array1[not_close_indices]
    elements_not_close_array2 = array2[not_close_indices]

    # Print the results
    print("Indices where elements are not close:", np.where(not_close_indices)[0])
    print("Elements in array1 not close:", elements_not_close_array1)
    print("Elements in array2 not close:", elements_not_close_array2)

    are_close = np.allclose(eigenvalues_cupy, eigenvalues_sklearn, atol=0, rtol=1e-5)
    print("Arrays are close:", are_close)
    array1 = eigenvalues_cupy
    array2 = eigenvalues_sklearn
    # Create a boolean array showing which elements are close
    close_mask = np.isclose(array1, array2, atol=0, rtol=1e-4)

    # Find the indices of elements that are not close
    not_close_indices = ~close_mask

    # Extract the elements from both arrays where they are not close
    elements_not_close_array1 = array1[not_close_indices]
    elements_not_close_array2 = array2[not_close_indices]

    # Print the results
    print("Indices where elements are not close:", np.where(not_close_indices)[0])
    print("Elements in array1 not close:", elements_not_close_array1)
    print("Elements in array2 not close:", elements_not_close_array2)

print()
print("raft vs cupy")
compare_arr(eigenvalues_raft, eigenvalues_cupy)
if X is not None:
    print("raft vs sklearn")
    compare_arr(eigenvalues_raft, eigenvalues_sklearn)
    print("cupy vs sklearn")
    compare_arr(eigenvalues_cupy, eigenvalues_sklearn)