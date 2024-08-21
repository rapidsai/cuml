import numpy as np
from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
import cupy as cp
from pylibraft.random import rmat

np.set_printoptions(linewidth=np.inf, precision=7, suppress=True, threshold=np.inf, floatmode='fixed')

r_scale = 3
c_scale = 3
sparsity = 1
n_edges = int(sparsity * (2**r_scale * 2**c_scale))
print("n_edges", n_edges)

theta_len = max(r_scale, c_scale) * 4

seed = 1234
cp.random.seed(seed)

out = cp.empty((n_edges * 100, 2), dtype=cp.int32)
theta = cp.random.random_sample(theta_len, dtype=cp.float32)

rmat(out, theta, r_scale, c_scale, seed=seed, handle=None)

# Extract source and destination node IDs
src_nodes = out[:, 0]
dst_nodes = out[:, 1]
data = cp.ones(src_nodes.shape[0])
# Compute the number of nodes
num_nodes = 2 ** max(r_scale, c_scale)

adj_matrix_csr = cucsr_matrix((data, (src_nodes, dst_nodes)), shape=(2*num_nodes, 2*num_nodes), dtype=cp.float32)

adj_matrix_csr = adj_matrix_csr + adj_matrix_csr.T
print(adj_matrix_csr.nnz, adj_matrix_csr.shape)