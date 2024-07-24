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

#pragma once

#include <stdint.h>

namespace raft {
class handle_t;
}

namespace ML {

namespace Spectral {

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
void fit_embedding(const raft::handle_t& handle,
                   int* rows,
                   int* cols,
                   float* vals,
                   int nnz,
                   int n,
                   int n_components,
                   float* out,
                   unsigned long long seed = 1234567);

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
                    int restartiter = 15);

struct SpectralParams {};

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
                  int nnz);

}  // namespace Spectral
}  // namespace ML
