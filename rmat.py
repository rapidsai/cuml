import numpy as np
from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
from scipy.sparse import csr_matrix
import cupy as cp
from pylibraft.random import rmat
import time
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import eigs, eigsh
from cupyx.scipy.sparse.linalg import eigsh as cueigsh
from cuml.manifold.lanczos import eig_lanczos

# np.set_printoptions(linewidth=np.inf, precision=7, suppress=True, threshold=np.inf, floatmode='fixed')
np.set_printoptions(linewidth=np.inf, suppress=True, threshold=np.inf)

r_scale = 12
c_scale = 12
sparsity = 1
n_edges = int(sparsity * (2**r_scale * 2**c_scale))
print("n_edges", n_edges)

theta_len = max(r_scale, c_scale) * 4

seed = 1234
cp.random.seed(seed)

out = cp.empty((n_edges, 2), dtype=cp.int32)
theta = cp.random.random_sample(theta_len, dtype=cp.float32)

rmat(out, theta, r_scale, c_scale, seed=seed, handle=None)

# Extract source and destination node IDs
src_nodes = out[:, 0]
dst_nodes = out[:, 1]
data = cp.ones(src_nodes.shape[0])

# Compute the number of nodes
num_nodes = 2 ** max(r_scale, c_scale)

adj_matrix_csr = cucsr_matrix((data, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes), dtype=cp.float32)
adj_matrix_csr = adj_matrix_csr + adj_matrix_csr.T
print(adj_matrix_csr.shape, adj_matrix_csr.nnz)

# raise ValueError("just testing")

def symmetric_normalize(csr):
    # Compute the degree matrix (sum of each row)
    degrees = csr.sum(axis=1).ravel()
    
    # Handle zero degrees by setting them to 1 to avoid division by zero
    degrees[degrees == 0] = 1
    
    # Compute the inverse square root of the degree matrix
    degrees_inv_sqrt = 1.0 / cp.sqrt(degrees)
    
    # Construct the inverse square root degree matrix
    degree_matrix_inv_sqrt = cp.sparse.diags(degrees_inv_sqrt)
    
    # Perform symmetric normalization
    normalized_csr = degree_matrix_inv_sqrt @ csr @ degree_matrix_inv_sqrt
    return normalized_csr


def cupy_to_scipy(cupy_csr):
    # Extract data, indices, and indptr from the cupy CSR matrix
    data = cp.asnumpy(cupy_csr.data)
    indices = cp.asnumpy(cupy_csr.indices)
    indptr = cp.asnumpy(cupy_csr.indptr)
    
    # Create a scipy CSR matrix using the extracted data
    scipy_csr = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)
    
    return scipy_csr


cuX = adj_matrix_csr
cuX = cuX + cuX.T
cuX = symmetric_normalize(cuX)
if (True):
    X = cupy_to_scipy(cuX)
    print(X.shape, X.nnz)
else:
    X = None

rng = cp.random.default_rng(42)
v0_dev = rng.random((num_nodes)).astype(cp.float32)

k=4000
n=num_nodes
ncv = None
if ncv is None:
    ncv = min(max(2 * k, k + 32), n - 1)
else:
    ncv = min(max(ncv, k + 2), n - 1)
ncv=ncv-k+1
print("ncv", ncv)


start = time.time()
eigenvalues_raft, eigenvectors, eig_iters = eig_lanczos(cuX, k, 42, dtype=np.float32, maxiter=10000, tol=1e-9, conv_n_iters=4, conv_eps=0.001, restartiter=ncv, v0=v0_dev, handle=None)
print("Raft Time (s)", time.time() - start)
print("eigenvalues", np.array2string(eigenvalues_raft, separator=', '), eig_iters)
print(eigenvectors[:, 0][:10])


start = time.time()
eigenvalues_cupy, eigenvectors = cueigsh(cuX, k=k, which="SA", maxiter=10000, tol=1e-9, v0=v0_dev)
print("Cupy eigsh Sklearn Time (s)", time.time() - start)
print("eigenvalues", np.array2string(eigenvalues_cupy, separator=', '))
print(eigenvectors[:, 0][:10])


if (X is not None):
    try:
        start = time.time()
        eigenvalues_sklearn, eigenvectors = eigsh((X), k=k, which="SA", maxiter=10000, tol=0)
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
#FIXME:
# eigenvalues_raft = eigenvalues_cupy
# eigenvalues_raft = cp.asnumpy(eigenvalues_raft)

eigenvalues_raft = np.asarray(eigenvalues_raft)
eigenvalues_cupy = cp.asnumpy(eigenvalues_cupy)


eigenvalues_raft[np.abs(eigenvalues_raft) < 1e-5] = 0
eigenvalues_sklearn[np.abs(eigenvalues_sklearn) < 1e-5] = 0
eigenvalues_cupy[np.abs(eigenvalues_cupy) < 1e-5] = 0

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


# eigenvalues, eigenvectors = np.linalg.eigh(X.todense(), UPLO='U')


# eigenvalues[np.abs(eigenvalues) < 1e-5] = 0
# print("numpy linalg eigh vs sklearn eigsh")
# compare_arr(eigenvalues[:k], eigenvalues_sklearn)
