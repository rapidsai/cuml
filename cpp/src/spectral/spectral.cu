/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuml/manifold/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/spectral.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda.h>

#include <memory>

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

/**
 * @brief Dimensionality reduction via TSNE using Barnes-Hut, Fourier Interpolation, or naive
 * methods. or brute force O(N^2).
 *
 * @param[in]  handle              The GPU handle.
 * @param[in]  X                   The row-major dataset in device memory.
 * @param[out] Y                   The column-major final embedding in device memory
 * @param[in]  n                   Number of rows in data X.
 * @param[in]  p                   Number of columns in data X.
 * @param[in]  knn_indices         Array containing nearest neighbors indices.
 * @param[in]  knn_dists           Array containing nearest neighbors distances.
 * @param[in]  params              Parameters for TSNE model
 * @param[out] kl_div              (optional) KL divergence output
 *
 * The CUDA implementation is derived from the excellent CannyLabs open source
 * implementation here: https://github.com/CannyLab/tsne-cuda/. The CannyLabs
 * code is licensed according to the conditions in
 * cuml/cpp/src/tsne/cannylabs_tsne_license.txt. A full description of their
 * approach is available in their article t-SNE-CUDA: GPU-Accelerated t-SNE and
 * its Applications to Modern Data (https://arxiv.org/abs/1807.11824).
 */
void spectral_fit(const raft::handle_t& handle,
                  float* X,
                  float* Y,
                  int n,
                  int p,
                  int* knn_indices,
                  float* knn_dists)
{
  manifold_dense_inputs_t<float> input(X, Y, n, p);
  knn_graph<int, float> k_graph(n, 90, knn_indices, knn_dists);

  std::unique_ptr<rmm::device_uvector<int>> knn_indices_b = nullptr;
  std::unique_ptr<rmm::device_uvector<float>> knn_dists_b = nullptr;
  auto stream                                             = handle.get_stream();

  knn_indices_b = std::make_unique<rmm::device_uvector<int>>(n * k_graph.n_neighbors, stream);
  knn_dists_b   = std::make_unique<rmm::device_uvector<float>>(n * k_graph.n_neighbors, stream);

  k_graph.knn_indices = knn_indices_b->data();
  k_graph.knn_dists   = knn_dists_b->data();

  std::vector<float*> input_vec = {input.X};
  std::vector<int> sizes_vec    = {input.n};

  raft::spatial::knn::brute_force_knn(handle,
                                      input_vec,
                                      sizes_vec,
                                      input.d,
                                      input.X,
                                      input.n,
                                      k_graph.knn_indices,
                                      k_graph.knn_dists,
                                      k_graph.n_neighbors,
                                      true,
                                      true,
                                      static_cast<std::vector<int>*>(nullptr),
                                      raft::distance::DistanceType::L2SqrtExpanded,
                                      2.0F);

  // k_graph.knn_indices = these are the column indices
  // k_graph.knn_dists   = these are the values (so what is the value of an edge, simply distance)
  // the row indices are start node
  // the col indices are dest node
  // the values are the edge weights

  std::vector<int> vec(n * k_graph.n_neighbors);
  for (uint64_t i = 0; i < vec.size(); i++) {
    vec[i] = i / k_graph.n_neighbors;
  }

  std::unique_ptr<rmm::device_uvector<int>> knn_rows;
  knn_rows = std::make_unique<rmm::device_uvector<int>>(n * k_graph.n_neighbors, stream);

  raft::copy(knn_rows->data(), vec.data(), n * k_graph.n_neighbors, stream);

  raft::sparse::COO<float, int> coo(stream, n * k_graph.n_neighbors, n, p);

  raft::sparse::linalg::symmetrize<int, float>(handle,
                                               knn_rows->data(),
                                               k_graph.knn_indices,
                                               k_graph.knn_dists,
                                               n,
                                               p,
                                               n * k_graph.n_neighbors,
                                               coo);

  Spectral::fit_embedding(
    handle, coo.rows(), coo.cols(), coo.vals(), coo.nnz, coo.n_rows, 2, Y, 1234);
}

}  // namespace Spectral
}  // namespace ML
