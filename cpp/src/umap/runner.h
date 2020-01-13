/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuml/manifold/umapparams.h>
#include "optimize.h"
#include "supervised.h"

#include "fuzzy_simpl_set/runner.h"
#include "init_embed/runner.h"
#include "knn_graph/runner.h"
#include "simpl_set_embed/runner.h"

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include "sparse/coo.h"
#include "sparse/csr.h"

#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <iostream>

namespace UMAPAlgo {

// Swap this as impls change for now.
namespace FuzzySimplSetImpl = FuzzySimplSet::Naive;
namespace SimplSetEmbedImpl = SimplSetEmbed::Algo;

using namespace ML;
using namespace MLCommon::Sparse;

template <int TPB_X, typename T>
__global__ void init_transform(int *indices, T *weights, int n,
                               const T *embeddings, int embeddings_n,
                               int n_components, T *result, int n_neighbors) {
  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  int i =
    row * n_neighbors;  // each thread processes one row of the dist matrix

  if (row < n) {
    for (int j = 0; j < n_neighbors; j++) {
      for (int d = 0; d < n_components; d++) {
        result[row * n_components + d] +=
          weights[i + j] * embeddings[indices[i + j] * n_components + d];
      }
    }
  }
}

/**
 * Fit exponential decay curve to find the parameters
 * a and b, which are based on min_dist and spread
 * parameters.
 */
void find_ab(UMAPParams *params, std::shared_ptr<deviceAllocator> d_alloc,
             cudaStream_t stream) {
  Optimize::find_params_ab(params, d_alloc, stream);
}

template <typename T, int TPB_X>
void _fit(const cumlHandle &handle,
          T *X,   // input matrix
          int n,  // rows
          int d,  // cols
          UMAPParams *params, T *embeddings) {
  cudaStream_t stream = handle.getStream();
  auto d_alloc = handle.getDeviceAllocator();

  int k = params->n_neighbors;

  if (params->verbose)
    std::cout << "n_neighbors=" << params->n_neighbors << std::endl;
  find_ab(params, d_alloc, stream);

  /**
   * Allocate workspace for kNN graph
   */
  MLCommon::device_buffer<int64_t> knn_indices(d_alloc, stream, n * k);
  MLCommon::device_buffer<T> knn_dists(d_alloc, stream, n * k);

  kNNGraph::run(X, n, X, n, d, knn_indices.data(), knn_dists.data(), k, params,
                d_alloc, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  COO<T> rgraph_coo(d_alloc, stream);

  FuzzySimplSet::run<TPB_X, T>(n, knn_indices.data(), knn_dists.data(), k,
                               &rgraph_coo, params, d_alloc, stream);

  /**
   * Remove zeros from simplicial set
   */
  COO<T> cgraph_coo(d_alloc, stream);
  MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&rgraph_coo, &cgraph_coo,
                                               d_alloc, stream);

  /**
   * Run initialization method
   */
  InitEmbed::run(handle, X, n, d, knn_indices.data(), knn_dists.data(),
                 &cgraph_coo, params, embeddings, stream, params->init);

  if (params->callback) {
    params->callback->setup<T>(n, params->n_components);
    params->callback->on_preprocess_end(embeddings);
  }

  /**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
  SimplSetEmbed::run<TPB_X, T>(X, n, d, &cgraph_coo, params, embeddings,
                               d_alloc, stream);

  if (params->callback) params->callback->on_train_end(embeddings);
}

template <typename T, int TPB_X>
void _fit(const cumlHandle &handle,
          T *X,  // input matrix
          T *y,  // labels
          int n, int d, UMAPParams *params, T *embeddings) {
  std::shared_ptr<deviceAllocator> d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  int k = params->n_neighbors;

  if (params->target_n_neighbors == -1)
    params->target_n_neighbors = params->n_neighbors;

  find_ab(params, d_alloc, stream);

  /**
   * Allocate workspace for kNN graph
   */
  MLCommon::device_buffer<int64_t> knn_indices(d_alloc, stream, n * k);
  MLCommon::device_buffer<T> knn_dists(d_alloc, stream, n * k);

  kNNGraph::run(X, n, X, n, d, knn_indices.data(), knn_dists.data(), k, params,
                d_alloc, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  /**
   * Allocate workspace for fuzzy simplicial set.
   */
  COO<T> rgraph_coo(d_alloc, stream);
  COO<T> tmp_coo(d_alloc, stream);

  /**
   * Run Fuzzy simplicial set
   */
  //int nnz = n*k*2;
  FuzzySimplSet::run<TPB_X, T>(n, knn_indices.data(), knn_dists.data(),
                               params->n_neighbors, &tmp_coo, params, d_alloc,
                               stream);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&tmp_coo, &rgraph_coo, d_alloc,
                                               stream);

  COO<T> final_coo(d_alloc, stream);

  /**
   * If target metric is 'categorical', perform
   * categorical simplicial set intersection.
   */
  if (params->target_metric == ML::UMAPParams::MetricType::CATEGORICAL) {
    if (params->verbose)
      std::cout << "Performing categorical intersection" << std::endl;
    Supervised::perform_categorical_intersection<TPB_X, T>(
      y, &rgraph_coo, &final_coo, params, d_alloc, stream);

    /**
     * Otherwise, perform general simplicial set intersection
     */
  } else {
    if (params->verbose)
      std::cout << "Performing general intersection" << std::endl;
    Supervised::perform_general_intersection<TPB_X, T>(
      handle, y, &rgraph_coo, &final_coo, params, stream);
  }

  /**
   * Remove zeros
   */
  MLCommon::Sparse::coo_sort<T>(&final_coo, d_alloc, stream);

  COO<T> ocoo(d_alloc, stream);
  MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&final_coo, &ocoo, d_alloc,
                                               stream);

  /**
   * Initialize embeddings
   */
  InitEmbed::run(handle, X, n, d, knn_indices.data(), knn_dists.data(), &ocoo,
                 params, embeddings, stream, params->init);

  if (params->callback) params->callback->on_preprocess_end(embeddings);

  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<TPB_X, T>(X, n, d, &ocoo, params, embeddings, d_alloc,
                               stream);

  if (params->callback) params->callback->on_train_end(embeddings);

  CUDA_CHECK(cudaPeekAtLastError());
}

/**
	 *
	 */
template <typename T, int TPB_X>
void _transform(const cumlHandle &handle, float *X, int n, int d, float *orig_X,
                int orig_n, T *embedding, int embedding_n, UMAPParams *params,
                T *transformed) {
  std::shared_ptr<deviceAllocator> d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  if (params->verbose) {
    std::cout << "Running transform" << std::endl;
  }

  if (params->verbose) {
    std::cout << "Building KNN Graph" << std::endl;
  }

  /**
   * Perform kNN of X
   */
  MLCommon::device_buffer<int64_t> knn_indices(d_alloc, stream,
                                               n * params->n_neighbors);
  MLCommon::device_buffer<T> knn_dists(d_alloc, stream,
                                       n * params->n_neighbors);

  kNNGraph::run(orig_X, orig_n, X, n, d, knn_indices.data(), knn_dists.data(),
                params->n_neighbors, params, d_alloc, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  float adjusted_local_connectivity =
    max(0.0, params->local_connectivity - 1.0);

  if (params->verbose) {
    std::cout << "Smoothing KNN distances" << std::endl;
  }

  /**
   * Perform smooth_knn_dist
   */
  MLCommon::device_buffer<T> sigmas(d_alloc, stream, n);
  MLCommon::device_buffer<T> rhos(d_alloc, stream, n);
  CUDA_CHECK(cudaMemsetAsync(sigmas.data(), 0, n * sizeof(T), stream));
  CUDA_CHECK(cudaMemsetAsync(rhos.data(), 0, n * sizeof(T), stream));

  dim3 grid_n(MLCommon::ceildiv(n, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  FuzzySimplSetImpl::smooth_knn_dist<TPB_X, T>(
    n, knn_indices.data(), knn_dists.data(), rhos.data(), sigmas.data(), params,
    params->n_neighbors, adjusted_local_connectivity, d_alloc, stream);

  /**
   * Compute graph of membership strengths
   */

  int nnz = n * params->n_neighbors;

  dim3 grid_nnz(MLCommon::ceildiv(nnz, TPB_X), 1, 1);

  if (params->verbose) {
    std::cout << "Executing fuzzy simplicial set" << std::endl;
  }

  /**
   * Allocate workspace for fuzzy simplicial set.
   */

  COO<T> graph_coo(d_alloc, stream, nnz, n, n);

  FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X>
    <<<grid_n, blk, 0, stream>>>(knn_indices.data(), knn_dists.data(),
                                 sigmas.data(), rhos.data(), graph_coo.vals(),
                                 graph_coo.rows(), graph_coo.cols(),
                                 graph_coo.n_rows, params->n_neighbors);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::device_buffer<int> row_ind(d_alloc, stream, n);
  MLCommon::device_buffer<int> ia(d_alloc, stream, n);

  MLCommon::Sparse::sorted_coo_to_csr(&graph_coo, row_ind.data(), d_alloc,
                                      stream);
  MLCommon::Sparse::coo_row_count<TPB_X>(&graph_coo, ia.data(), stream);

  MLCommon::device_buffer<T> vals_normed(d_alloc, stream, graph_coo.nnz);
  CUDA_CHECK(
    cudaMemsetAsync(vals_normed.data(), 0, graph_coo.nnz * sizeof(T), stream));

  if (params->verbose) {
    std::cout << "Performing L1 normalization" << std::endl;
  }

  MLCommon::Sparse::csr_row_normalize_l1<TPB_X, T>(
    row_ind.data(), graph_coo.vals(), graph_coo.nnz, graph_coo.n_rows,
    vals_normed.data(), stream);

  init_transform<TPB_X, T><<<grid_n, blk, 0, stream>>>(
    graph_coo.cols(), vals_normed.data(), graph_coo.n_rows, embedding,
    embedding_n, params->n_components, transformed, params->n_neighbors);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaMemsetAsync(ia.data(), 0.0, ia.size() * sizeof(int), stream));

  CUDA_CHECK(cudaPeekAtLastError());

  /**
   * Go through COO values and set everything that's less than
   * vals.max() / params->n_epochs to 0.0
   */
  thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(graph_coo.vals());
  T max =
    *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + nnz));

  int n_epochs = params->n_epochs;
  if (params->n_epochs <= 0) {
    if (n <= 10000)
      n_epochs = 100;
    else
      n_epochs = 30;
  } else {
    n_epochs /= 3;
  }

  if (params->verbose) {
    std::cout << "n_epochs=" << n_epochs << std::endl;
  }

  MLCommon::LinAlg::unaryOp<T>(
    graph_coo.vals(), graph_coo.vals(), graph_coo.nnz,
    [=] __device__(T input) {
      if (input < (max / float(n_epochs)))
        return 0.0f;
      else
        return input;
    },
    stream);

  CUDA_CHECK(cudaPeekAtLastError());

  /**
   * Remove zeros
   */
  MLCommon::Sparse::COO<T> comp_coo(d_alloc, stream);
  MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&graph_coo, &comp_coo, d_alloc,
                                               stream);

  if (params->verbose) {
    std::cout << "Computing # of epochs for training each sample" << std::endl;
  }

  MLCommon::device_buffer<T> epochs_per_sample(d_alloc, stream, nnz);

  SimplSetEmbedImpl::make_epochs_per_sample(
    comp_coo.vals(), comp_coo.nnz, n_epochs, epochs_per_sample.data(), stream);

  if (params->verbose) {
    std::cout << "Performing optimization" << std::endl;
  }

  SimplSetEmbedImpl::optimize_layout<TPB_X, T>(
    transformed, n, embedding, embedding_n, comp_coo.rows(), comp_coo.cols(),
    comp_coo.nnz, epochs_per_sample.data(), n, params->repulsion_strength,
    params, n_epochs, d_alloc, stream);
}

}  // namespace UMAPAlgo
