import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from cuml.manifold.lanczos import eig_lanczos
from scipy.sparse.linalg import lobpcg
from scipy.sparse import random as sparse_random
from scipy.sparse import rand as sparse_rand
from scipy.sparse.linalg import eigs, eigsh
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy import sparse
import time
from sklearn.manifold import SpectralEmbedding
from cuml.manifold import SpectralEmbedding as cuSpectralEmbedding
from scipy.sparse.linalg import ArpackError
from cupyx.scipy.sparse.linalg import eigsh as cueigsh
from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
from cupyx.scipy.sparse import random as cusparse_random
import cupy as cp


# np.set_printoptions(precision=4, suppress=True, threshold=50)
# np.printoptions(precision=5, suppress=True, threshold=np.inf)
np.set_printoptions(linewidth=np.inf, precision=7, suppress=True, threshold=np.inf, floatmode='fixed')
# np.set_printoptions(linewidth=np.inf, suppress=True, threshold=np.inf)

import numpy as np
import cupy as cp
from pylibraft.random import rmat
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, diags


r_scale = 10
c_scale = 10
sparsity = 1
n_edges = int(sparsity * (2**r_scale * 2**c_scale))
print(n_edges)

theta_len = max(r_scale, c_scale) * 4

seed = 1234
cp.random.seed(seed)

out = cp.empty((n_edges, 2), dtype=cp.int32)
theta = cp.random.random_sample(theta_len, dtype=cp.float32)

rmat(out, theta, r_scale, c_scale, seed=seed, handle=None)

# time.sleep(10)

# raise ValueError("yo")

# Convert CuPy array to NumPy array
# out_np = cp.asnumpy(out)

#out_np = np.unique(out_np, axis=0)


# Extract source and destination node IDs
src_nodes = out[:, 0]
dst_nodes = out[:, 1]
data = cp.ones(src_nodes.shape[0])
# print(src_nodes)
# print(dst_nodes)
print(src_nodes.shape)
print(dst_nodes.shape)
print(data.shape)

# Compute the number of nodes
num_nodes = 2 ** max(r_scale, c_scale)


# # Create histogram of node degrees
# degree_counts = cp.bincount(src_nodes, minlength=num_nodes)

# # Calculate indptr
# indptr = cp.zeros(num_nodes + 1, dtype=cp.int32)
# indptr[1:] = cp.cumsum(degree_counts)

# # Calculate indices
# indices = cp.zeros(n_edges, dtype=cp.int32)
# indices = dst_nodes

# # Create the CSR matrix using CuPy
# adj_matrix_csr = cucsr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=cp.float32)

# raise ValueError("yo")

# Create a CSR matrix directly
# from cupyx.scipy.sparse import coo_matrix

# adj_matrix_coo = coo_matrix((data, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes), dtype=cp.float32)
# test = adj_matrix_coo.tocsr()

adj_matrix_csr = cucsr_matrix((data, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes), dtype=cp.float32)
adj_matrix_csr = adj_matrix_csr + adj_matrix_csr.T
print(adj_matrix_csr.nnz, adj_matrix_csr.shape)
# print(adj_matrix_csr.todense())



raise ValueError("just testing")


# # Print out the sparse adjacency matrix (for debugging purposes)

# def symmetric_normalize(csr):
#     # Compute the degree matrix (sum of each row)
#     degrees = csr.sum(axis=1).ravel()
    
#     # Handle zero degrees by setting them to 1 to avoid division by zero
#     degrees[degrees == 0] = 1
    
#     # Compute the inverse square root of the degree matrix
#     degrees_inv_sqrt = 1.0 / cp.sqrt(degrees)
    
#     # Construct the inverse square root degree matrix
#     degree_matrix_inv_sqrt = cp.sparse.diags(degrees_inv_sqrt)
    
#     # Perform symmetric normalization
#     normalized_csr = degree_matrix_inv_sqrt @ csr @ degree_matrix_inv_sqrt
#     return normalized_csr

# n_samples = num_nodes
# n_components = 2000

# X = adj_matrix_csr

# X = X + X.T

# X = symmetric_normalize(X)
# # cuX = cucsr_matrix(X)
# cuX = X
# rng = cp.random.default_rng(42)
# v0_dev = rng.random((n_samples)).astype(cp.float32)
# # v0_dev = cp.asarray(v0)

# print(X.nnz, X.shape)

# def cupy_to_scipy(cupy_csr):
#     # Extract data, indices, and indptr from the cupy CSR matrix
#     data = cp.asnumpy(cupy_csr.data)
#     indices = cp.asnumpy(cupy_csr.indices)
#     indptr = cp.asnumpy(cupy_csr.indptr)
    
#     # Create a scipy CSR matrix using the extracted data
#     scipy_csr = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes), dtype=np.float32)
    
#     return scipy_csr



# # test=[[4,2], [2, 3]]
# # test = cp.asarray(test, dtype=cp.float32)
# # cuX = cucsr_matrix(test)

# if (True):
#     X = cupy_to_scipy(cuX)
#     print(X.shape, X.nnz)
# else:
#     X = None

# print(cuX.dtype)

# print("dense input", np.array2string(cuX.toarray(), separator=', '))

# # Compute the degree of each row
# row_degrees = cp.diff(cuX.indptr)

# # Print the degree of each row
# for i, degree in enumerate(row_degrees):
#     print(f"Row {i} degree: {degree}")

# raise ValueError("yo")


# ----#------

n_samples = 10000
n_features = 10000
n_components = 2000
n_neighbors = 70
sparsity = 0.01
start = time.time()
rng = np.random.default_rng(42)
X = sparse_random(n_samples, n_features, density=sparsity, random_state=rng, dtype=np.float32)
X = X + X.T
print(X.nnz, X.shape)

cuX = cucsr_matrix(X)
v0 = rng.random((n_samples)).astype(np.float32)
v0_dev = cp.asarray(v0)
# raise ValueError("just testing")
# print(X.nnz, X.shape)
# print(cuX.nnz, cuX.shape)
# print("indices", np.array2string(X.indices, separator=', '))
# print("indptr", np.array2string(X.indptr, separator=', '))
# print("data", np.array2string(X.data, separator=', '))
# print("v0", np.array2string(v0, separator=', '))


ncv = None
k=n_components
n=n_samples
if ncv is None:
    ncv = min(max(2 * k, k + 32), n - 1)
else:
    ncv = min(max(ncv, k + 2), n - 1)
ncv=ncv-k
print("ncv", ncv)




# cuX = cusparse_random(n_samples, n_features, density=sparsity, random_state=42, dtype=np.float32)
# cuX = cuX + cuX.T

# X = X.astype(np.float32)
# X = X + X.T
# X = kneighbors_graph(
#     X, n_neighbors, include_self=True, mode="distance",
# )
# X = X + X.T
# matrix1 = X

# laplacian, dd = csgraph_laplacian(
#     X, normed=False, return_diag=True
# )
# laplacian = _set_diag(laplacian, 1, False)
# # X = laplacian + laplacian.T
# X = laplacian

# print(X)

# import rmm
# pool = rmm.mr.PoolMemoryResource(
#     rmm.mr.CudaMemoryResource(),
#     initial_pool_size=3*2**32,
#     maximum_pool_size=3*2**32,
# )
# rmm.mr.set_current_device_resource(pool)

# import rmm
# import cupy as cp
# rmm.reinitialize(
#     pool_allocator=True,
#     initial_pool_size=6 * 1024**3  # 6 GB
# )
# # Define the size of the array in terms of elements
# num_elements = 6 * 1024**3 // 4  # Assuming float32 (4 bytes per element)

# # Allocate GPU memory using CuPy
# gpu_array = cp.zeros(num_elements, dtype=cp.float32)
# time.sleep(10)
# raise ValueError()

# r = np.random.RandomState(42)
# X0 = r.standard_normal(size=(n_samples, n_components + 1))

# start = time.time()
# eigenvalues, eigenvectors = lobpcg((X), X=X0, tol=1e-9, largest=False, maxiter=10000)
# print("Lobpcg Sklearn Time (s)", time.time() - start)
# print(eigenvalues)



start = time.time()
eigenvalues_raft, eigenvectors, eig_iters = eig_lanczos(cuX, n_components, 42, dtype=np.float32, maxiter=10000, tol=1e-15, conv_n_iters=4, conv_eps=0.001, restartiter=ncv, v0=v0_dev, handle=None)
print("Raft Time (s)", time.time() - start)
print("eigenvalues", np.array2string(eigenvalues_raft, separator=', '), eig_iters)
print(eigenvectors[:, 0][:10])


start = time.time()
eigenvalues_cupy, eigenvectors = cueigsh(cuX, k=n_components, which="SA", maxiter=10000, tol=1e-9, v0=v0_dev)
print("Cupy eigsh Sklearn Time (s)", time.time() - start)
print("eigenvalues", np.array2string(eigenvalues_cupy, separator=', '))
print(eigenvectors[:, 0][:10])


if (X is not None):
    try:
        start = time.time()
        eigenvalues_sklearn, eigenvectors = eigsh((X), k=n_components, which="SA", maxiter=10000, tol=0)
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

are_close = np.allclose(eigenvalues_raft, eigenvalues_cupy, atol=0)
print("Arrays are close:", are_close)
if X is not None:
    are_close = np.allclose(eigenvalues_raft, eigenvalues_sklearn, atol=0)
    print("Arrays are close:", are_close)

print()
print("raft vs cupy")
compare_arr(eigenvalues_raft, eigenvalues_cupy)
if X is not None:
    print("raft vs sklearn")
    compare_arr(eigenvalues_raft, eigenvalues_sklearn)
    print("cupy vs sklearn")
    compare_arr(eigenvalues_cupy, eigenvalues_sklearn)


# print(np.allclose(eigenvalues1, eigenvalues, atol=1e-3))

# dot1_double, dot2_double, dot3_double = np.dot(eigenvectors[:, 0], eigenvectors[:, 1]), np.dot(eigenvectors[:, 0], eigenvectors[:, 2]), np.dot(eigenvectors[:, 1], eigenvectors[:, 2])
# print(dot1_double, dot2_double, dot3_double)


# start = time.time()
# s = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=42, eigen_solver="lobpcg")
# s.affinity_matrix_ = matrix1
# out = s.fit_transform(matrix1)
# print("SpectralEmbedding Time (s)", time.time() - start)

# matrix1 = matrix1.tocoo()
# start = time.time()
# s = cuSpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=42, rows=matrix1.row, cols=matrix1.col, vals=matrix1.data, nnz=len(matrix1.row))
# out = s.fit_transform(matrix1.toarray())
# print("Raft SpectralEmbedding Time (s)", time.time() - start)

