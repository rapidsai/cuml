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

struct COOMatrix {
  std::vector<int> row;
  std::vector<int> col;
  std::vector<float> data;
};

// Function to symmetrize, sort, and remove zeros from a COO matrix
COOMatrix transpose(const COOMatrix& matrix)
{
  COOMatrix res;
  res.data = matrix.data;
  for (int i = 0; i < matrix.row.size(); i++) {
    res.row.push_back(matrix.col[i]);
    res.col.push_back(matrix.row[i]);
  }
  return res;
}

// Function to symmetrize, sort, and remove zeros from a COO matrix
COOMatrix add(const COOMatrix& matrix1, const COOMatrix& matrix2)
{
  COOMatrix res;
  std::unordered_map<int, std::unordered_map<int, float>> hashMap;
  auto populate = [&](const COOMatrix& matrix) {
    for (int i = 0; i < matrix.row.size(); i++) {
      if (hashMap.count(matrix.row[i]) <= 0) {
        hashMap[matrix.row[i]] = {{matrix.col[i], matrix.data[i]}};
      } else {
        if (hashMap[matrix.row[i]].count(matrix.col[i]) <= 0) {
          hashMap[matrix.row[i]][matrix.col[i]] = matrix.data[i];
        } else {
          hashMap[matrix.row[i]][matrix.col[i]] += matrix.data[i];
        }
      }
    }
  };
  populate(matrix1);
  populate(matrix2);
  for (auto& pair : hashMap) {
    int row = pair.first;
    for (auto& cols : pair.second) {
      int col   = cols.first;
      float val = cols.second;
      res.row.push_back(row);
      res.col.push_back(col);
      res.data.push_back(val);
    }
  }

  return res;
}

// Function to sort COOMatrix by row and then by col
void sortCOOMatrix(COOMatrix& mat)
{
  std::vector<int> idx(mat.row.size());
  std::iota(idx.begin(), idx.end(), 0);  // Initialize index vector with 0, 1, ..., n-1

  // Sort by row and then by col
  std::sort(idx.begin(), idx.end(), [&mat](int i, int j) {
    if (mat.row[i] != mat.row[j]) {
      return mat.row[i] < mat.row[j];
    } else {
      return mat.col[i] < mat.col[j];
    }
  });

  // Rearrange row, col, and data vectors according to idx
  std::vector<int> sorted_row(mat.row.size());
  std::vector<int> sorted_col(mat.col.size());
  std::vector<float> sorted_data(mat.data.size());

  for (size_t i = 0; i < idx.size(); ++i) {
    sorted_row[i]  = mat.row[idx[i]];
    sorted_col[i]  = mat.col[idx[i]];
    sorted_data[i] = mat.data[idx[i]];
  }

  // Assign sorted values back to the original matrix
  mat.row  = sorted_row;
  mat.col  = sorted_col;
  mat.data = sorted_data;
}

void removeZeros(COOMatrix& mat)
{
  std::vector<int> new_row;
  std::vector<int> new_col;
  std::vector<float> new_data;

  for (size_t i = 0; i < mat.data.size(); ++i) {
    if (mat.data[i] != 0.0) {
      new_row.push_back(mat.row[i]);
      new_col.push_back(mat.col[i]);
      new_data.push_back(mat.data[i]);
    }
  }

  // Update COOMatrix with non-zero values
  mat.row  = new_row;
  mat.col  = new_col;
  mat.data = new_data;
}

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
 * Given a COO formatted (symmetric) knn graph, this function
 * computes the spectral embeddings (lowest n_components
 * eigenvectors), using Lanczos min cut algorithm.
 * @param handle cuml handle
 * @param rows source vertices of knn graph (size nnz)
 * @param cols destination vertices of knn graph (size nnz)
 * @param vals edge weights connecting vertices of knn graph (size nnz)
 * @param nnz size of rows/cols/vals
 * @param n number of samples in X
 * @param n_components the number of components to project the X into
 * @param out output array for embedding (size n*n_comonents)
 * @param seed random seed to use in both the lanczos solver and k-means
 */
template <typename T>
void lanczos_solver(const raft::handle_t& handle,
                    int* rows,
                    int* cols,
                    T* vals,
                    int nnz,
                    int n,
                    int n_components,
                    T* eigenvectors,
                    T* eigenvalues,
                    int* eig_iters,
                    unsigned long long seed = 1234567,
                    int maxiter             = 4000,
                    float tol               = 0.01,
                    int conv_n_iters        = 5,
                    float conv_eps          = 0.001,
                    int restartiter = 15)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<int> src_offsets(n + 1, stream);
  rmm::device_uvector<int> dst_cols(nnz, stream);
  rmm::device_uvector<T> dst_vals(nnz, stream);
  raft::sparse::convert::coo_to_csr(
    handle, rows, cols, vals, nnz, n, src_offsets.data(), dst_cols.data(), dst_vals.data());

  rmm::device_uvector<T> eigVals(n_components, stream);
  rmm::device_uvector<T> eigVecs(n * (n_components), stream);
  rmm::device_uvector<int> labels(n, stream);

  raft::resource::sync_stream(handle, stream);

  using index_type = int;
  using value_type = T;

  index_type* ro = src_offsets.data();
  index_type* ci = dst_cols.data();
  value_type* vs = dst_vals.data();

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const csr_m{
    handle, ro, ci, vs, n, nnz};

  index_type neigvs = n_components;  // DO NOT USE + 1 here
  // index_type maxiter      = 4000;  // default reset value (when set to 0);
  // value_type tol          = 0.01;
  index_type restart_iter = restartiter + neigvs;  // what cugraph is using //FIXME:
  // int conv_n_iters = 5;
  // float conv_eps = 0.001;

  raft::spectral::eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol};

  cfg.seed         = seed;
  cfg.conv_n_iters = conv_n_iters;
  cfg.conv_eps     = conv_eps;

  raft::spectral::lanczos_solver_t<index_type, value_type> eigen_solver{cfg};

  using vertex_t = index_type;
  using weight_t = value_type;

  std::tuple<vertex_t, weight_t, vertex_t>
    stats;  //{iters_eig_solver,residual_cluster,iters_cluster_solver} // # iters eigen solver,
            // cluster solver residual, # iters cluster solver

  // Compute smallest eigenvalues and eigenvectors
  std::get<0>(stats) =
    eigen_solver.solve_smallest_eigenvectors(handle, csr_m, eigVals.data(), eigVecs.data());
  // std::cout << "stats" << std::get<0>(stats) << std::endl;
  raft::copy<int>(eig_iters, &std::get<0>(stats), 1, stream);

  raft::copy<T>(eigenvectors, eigVecs.data(), n * n_components, stream);
  raft::copy<T>(eigenvalues, eigVals.data(), n_components, stream);

  std::ofstream out_file("output.txt");
  if (!out_file.is_open()) { std::cerr << "Failed to open output file!" << std::endl; }
  raft::print_device_vector("eigenvals", eigVals.data(), n_components, out_file);
  raft::print_device_vector("eigenvecs", eigVecs.data(), n * (n_components), out_file);
  raft::print_device_vector("eigenvectors", eigenvectors, n * n_components, out_file);
  raft::print_device_vector("eigenvalues", eigenvalues, n_components, out_file);

  RAFT_CUDA_TRY(cudaGetLastError());
}

void lanczos_solver(const raft::handle_t& handle,
                    int* rows,
                    int* cols,
                    float* vals,
                    int nnz,
                    int n,
                    int n_components,
                    float* eigenvectors,
                    float* eigenvalues,
                    int* eig_iters,
                    unsigned long long seed = 1234567,
                    int maxiter             = 4000,
                    float tol               = 0.01,
                    int conv_n_iters        = 5,
                    float conv_eps          = 0.001,
                    int restartiter = 15)
{
  lanczos_solver<float>(handle,
                        rows,
                        cols,
                        vals,
                        nnz,
                        n,
                        n,
                        eigenvectors,
                        eigenvalues,
                        eig_iters,
                        seed,
                        maxiter,
                        tol,
                        conv_n_iters,
                        conv_eps,
                        restartiter);
}

void lanczos_solver(const raft::handle_t& handle,
                    int* rows,
                    int* cols,
                    double* vals,
                    int nnz,
                    int n,
                    int n_components,
                    double* eigenvectors,
                    double* eigenvalues,
                    int* eig_iters,
                    unsigned long long seed = 1234567,
                    int maxiter             = 4000,
                    float tol               = 0.01,
                    int conv_n_iters        = 5,
                    float conv_eps          = 0.001,
                    int restartiter = 15)
{
  lanczos_solver<double>(handle,
                         rows,
                         cols,
                         vals,
                         nnz,
                         n,
                         n,
                         eigenvectors,
                         eigenvalues,
                         eig_iters,
                         seed,
                         maxiter,
                         tol,
                         conv_n_iters,
                         conv_eps,
                         restartiter);
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
                  int* knn_rows,
                  float* knn_dists,
                  int* a_knn_indices,
                  int* a_knn_rows,
                  float* a_knn_dists,
                  int num_neighbors,
                  int* rows,
                  int* cols,
                  float* vals,
                  int nnz)
{
  manifold_dense_inputs_t<float> input(X, Y, n, p);
  knn_graph<int, float> k_graph(n, num_neighbors, knn_indices, knn_dists);

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

  // raft::copy(knn_dists, k_graph.knn_dists, n * k_graph.n_neighbors, stream);
  // raft::copy(knn_indices, k_graph.knn_indices, n * k_graph.n_neighbors, stream);

  // std::cout << k_graph.knn_dists << std::endl;
  // std::cout << k_graph.knn_indices << std::endl;

  // std::cout << knn_dists << std::endl;
  // std::cout << knn_indices << std::endl;

  // raft::print_device_vector("knn_indices", k_graph.knn_indices, n*k_graph.n_neighbors,
  // std::cout); raft::print_device_vector("knn_dists", k_graph.knn_dists, n*k_graph.n_neighbors,
  // std::cout);

  // k_graph.knn_indices = these are the column indices
  // k_graph.knn_dists   = these are the values (so what is the value of an edge, simply distance)
  // the row indices are start node
  // the col indices are dest node
  // the values are the edge weights

  std::vector<int> vec(n * k_graph.n_neighbors);
  for (uint64_t i = 0; i < vec.size(); i++) {
    vec[i] = i / k_graph.n_neighbors;
  }

  std::unique_ptr<rmm::device_uvector<int>> d_knn_rows;
  d_knn_rows = std::make_unique<rmm::device_uvector<int>>(n * k_graph.n_neighbors, stream);

  raft::copy(d_knn_rows->data(), vec.data(), n * k_graph.n_neighbors, stream);

  raft::sparse::COO<float, int> coo(stream, n * k_graph.n_neighbors, n, p);

  std::cout << "new" << std::endl;
  std::vector<int> coo_cols(n * k_graph.n_neighbors);
  std::vector<float> coo_values(n * k_graph.n_neighbors);
  std::vector<int> coo_rows = vec;

  raft::copy(coo_cols.data(), k_graph.knn_indices, n * k_graph.n_neighbors, stream);
  std::cout << "new1" << std::endl;
  raft::copy(coo_values.data(), k_graph.knn_dists, n * k_graph.n_neighbors, stream);
  std::cout << "new2" << std::endl;

  COOMatrix coo_in{coo_rows, coo_cols, coo_values};
  COOMatrix coo_transpose = transpose(coo_in);
  COOMatrix coo_symmetric = add(coo_in, coo_transpose);
  removeZeros(coo_symmetric);
  sortCOOMatrix(coo_symmetric);

  std::cout << "data size: " << coo_symmetric.data.size() << std::endl;
  std::cout << "col size: " << coo_symmetric.col.size() << std::endl;
  std::cout << "row size: " << coo_symmetric.row.size() << std::endl;
  std::cout << "n: " << n << std::endl;
  std::cout << "k_graph.n_neighbors: " << k_graph.n_neighbors << std::endl;

  raft::copy(knn_dists, coo_symmetric.data.data(), coo_symmetric.data.size(), stream);
  raft::copy(knn_indices, coo_symmetric.col.data(), coo_symmetric.col.size(), stream);
  raft::copy(knn_rows, coo_symmetric.row.data(), coo_symmetric.row.size(), stream);
  //   raft::copy(a_knn_dists, coo_in.data.data(), n * k_graph.n_neighbors, stream);
  //   raft::copy(a_knn_indices, coo_in.col.data(), n * k_graph.n_neighbors, stream);
  //   raft::copy(a_knn_rows, coo_in.row.data(), n * k_graph.n_neighbors, stream);

  raft::sparse::linalg::symmetrize<int, float>(handle,
                                               d_knn_rows->data(),
                                               k_graph.knn_indices,
                                               k_graph.knn_dists,
                                               n,
                                               p,
                                               n * k_graph.n_neighbors,
                                               coo);

  // raft::sparse::linalg::from_knn_symmetrize_matrix(k_graph.knn_indices, k_graph.knn_dists, n,
  // k_graph.n_neighbors, &coo, stream);

  raft::copy(a_knn_dists, coo.vals(), n * k_graph.n_neighbors, stream);
  raft::copy(a_knn_indices, coo.cols(), n * k_graph.n_neighbors, stream);
  raft::copy(a_knn_rows, coo.rows(), n * k_graph.n_neighbors, stream);

  // raft::sparse::linalg::coo_symmetrize<float>(
  //   &coo_in,
  //   &coo,
  //   [] __device__(int row, int col, float result, float transpose) {
  //     return 0.5 * (result + transpose);
  //   },
  //   stream);

  // raft::copy(knn_dists, coo.vals(), n * k_graph.n_neighbors, stream);
  // raft::copy(knn_indices, coo.cols(), n * k_graph.n_neighbors, stream);

  Spectral::fit_embedding(handle, rows, cols, vals, nnz, n, 2, Y, 1234);
}

}  // namespace Spectral
}  // namespace ML
