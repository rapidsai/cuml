/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/handle.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/spectral.cuh>

namespace raft {
class handle_t;
}

namespace ML {

namespace Spectral {

/**
 * Given a COO formatted (symmetric) knn graph, this function
 * computes the spectral embeddings (lowest n_components
 * eigenvectors), using Lanczos min cut algorithm.
 * @param rows source vertices of knn graph (size nnz)
 * @param cols destination vertices of knn graph (size nnz)
 * @param vals edge weights connecting vertices of knn graph (size nnz)
 * @param nnz size of rows/cols/vals
 * @param n number of samples in X
 * @param n_neighbors the number of neighbors to query for knn graph construction
 * @param n_components the number of components to project the X into
 * @param out output array for embedding (size n*n_comonents)
 */
void fit_embedding(const raft::handle_t& handle,
                   int* rows,
                   int* cols,
                   float* vals,
                   int nnz,
                   int n,
                   int n_components,
                   float* out,
                   unsigned long long seed)
{
  raft::sparse::spectral::fit_embedding(handle, rows, cols, vals, nnz, n, n_components, out, seed);
}
}  // namespace Spectral
}  // namespace ML
