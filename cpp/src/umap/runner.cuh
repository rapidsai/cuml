/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "optimize.cuh"
#include "supervised.cuh"
#include <cuml/common/logger.hpp>
#include <cuml/manifold/common.hpp>
#include <cuml/manifold/umapparams.h>

#include "fuzzy_simpl_set/runner.cuh"
#include "init_embed/runner.cuh"
#include "knn_graph/runner.cuh"
#include "simpl_set_embed/runner.cuh"

#include <memory>

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>

#include <raft/cuda_utils.cuh>

#include <common/nvtx.hpp>
#include <cuda_runtime.h>
#include <raft/core/nvtx.hpp>

namespace UMAPAlgo {

// Swap this as impls change for now.
namespace FuzzySimplSetImpl = FuzzySimplSet::Naive;
namespace SimplSetEmbedImpl = SimplSetEmbed::Algo;

using namespace ML;

template <int TPB_X, typename T>
__global__ void init_transform(int* indices,
                               T* weights,
                               int n,
                               const T* embeddings,
                               int embeddings_n,
                               int n_components,
                               T* result,
                               int n_neighbors)
{
  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  int i   = row * n_neighbors;  // each thread processes one row of the dist matrix

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
inline void find_ab(UMAPParams* params, cudaStream_t stream)
{
  Optimize::find_params_ab(params, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, int TPB_X>
void _get_graph(const raft::handle_t& handle,
                const umap_inputs& inputs,
                UMAPParams* params,
                raft::sparse::COO<value_t, int>* graph)
{
  raft::common::nvtx::range fun_scope("umap::supervised::_get_graph");
  cudaStream_t stream = handle.get_stream();

  int k = params->n_neighbors;

  ML::Logger::get().setLevel(params->verbosity);

  CUML_LOG_DEBUG("n_neighbors=%d", params->n_neighbors);

  raft::common::nvtx::push_range("umap::knnGraph");
  std::unique_ptr<rmm::device_uvector<value_idx>> knn_indices_b = nullptr;
  std::unique_ptr<rmm::device_uvector<value_t>> knn_dists_b     = nullptr;

  knn_graph<value_idx, value_t> knn_graph(inputs.n, k);

  /**
   * If not given precomputed knn graph, compute it
   */
  if (inputs.alloc_knn_graph()) {
    /**
     * Allocate workspace for kNN graph
     */
    knn_indices_b = std::make_unique<rmm::device_uvector<value_idx>>(inputs.n * k, stream);
    knn_dists_b   = std::make_unique<rmm::device_uvector<value_t>>(inputs.n * k, stream);

    knn_graph.knn_indices = knn_indices_b->data();
    knn_graph.knn_dists   = knn_dists_b->data();
  }

  CUML_LOG_DEBUG("Calling knn graph run");

  kNNGraph::run<value_idx, value_t, umap_inputs>(
    handle, inputs, inputs, knn_graph, k, params, stream);
  raft::common::nvtx::pop_range();

  CUML_LOG_DEBUG("Done. Calling fuzzy simplicial set");

  raft::common::nvtx::push_range("umap::simplicial_set");
  raft::sparse::COO<value_t> fss_graph(stream);
  FuzzySimplSet::run<TPB_X, value_idx, value_t>(
    inputs.n, knn_graph.knn_indices, knn_graph.knn_dists, k, &fss_graph, params, stream);

  CUML_LOG_DEBUG("Done. Calling remove zeros");

  /**
   * Remove zeros from simplicial set
   */
  raft::sparse::op::coo_remove_zeros<value_t>(&fss_graph, graph, stream);
  raft::common::nvtx::pop_range();
}

template <typename value_idx, typename value_t, typename umap_inputs, int TPB_X>
void _get_graph_supervised(const raft::handle_t& handle,
                           const umap_inputs& inputs,
                           UMAPParams* params,
                           raft::sparse::COO<value_t, int>* graph)
{
  raft::common::nvtx::range fun_scope("umap::supervised::_get_graph_supervised");
  cudaStream_t stream = handle.get_stream();

  int k = params->n_neighbors;

  ML::Logger::get().setLevel(params->verbosity);

  if (params->target_n_neighbors == -1) params->target_n_neighbors = params->n_neighbors;

  raft::common::nvtx::push_range("umap::knnGraph");
  std::unique_ptr<rmm::device_uvector<value_idx>> knn_indices_b = nullptr;
  std::unique_ptr<rmm::device_uvector<value_t>> knn_dists_b     = nullptr;

  knn_graph<value_idx, value_t> knn_graph(inputs.n, k);

  /**
   * If not given precomputed knn graph, compute it
   */
  if (inputs.alloc_knn_graph()) {
    /**
     * Allocate workspace for kNN graph
     */
    knn_indices_b = std::make_unique<rmm::device_uvector<value_idx>>(inputs.n * k, stream);
    knn_dists_b   = std::make_unique<rmm::device_uvector<value_t>>(inputs.n * k, stream);

    knn_graph.knn_indices = knn_indices_b->data();
    knn_graph.knn_dists   = knn_dists_b->data();
  }

  kNNGraph::run<value_idx, value_t, umap_inputs>(
    handle, inputs, inputs, knn_graph, k, params, stream);

  raft::common::nvtx::pop_range();

  /**
   * Allocate workspace for fuzzy simplicial set.
   */
  raft::common::nvtx::push_range("umap::simplicial_set");

  raft::sparse::COO<value_t> fss_graph(stream);
  {
    raft::sparse::COO<value_t> fss_graph_tmp(stream);
    /**
     * Run Fuzzy simplicial set
     */
    // int nnz = n*k*2;
    FuzzySimplSet::run<TPB_X, value_idx, value_t>(inputs.n,
                                                  knn_graph.knn_indices,
                                                  knn_graph.knn_dists,
                                                  params->n_neighbors,
                                                  &fss_graph_tmp,
                                                  params,
                                                  stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    raft::sparse::op::coo_remove_zeros<value_t>(&fss_graph_tmp, &fss_graph, stream);
  }

  raft::sparse::COO<value_t> ci_graph(stream);
  /**
   * If target metric is 'categorical', perform
   * categorical simplicial set intersection.
   */
  if (params->target_metric == ML::UMAPParams::MetricType::CATEGORICAL) {
    CUML_LOG_DEBUG("Performing categorical intersection");
    Supervised::perform_categorical_intersection<TPB_X, value_t>(
      inputs.y, &fss_graph, &ci_graph, params, stream);

    /**
     * Otherwise, perform general simplicial set intersection
     */
  } else {
    CUML_LOG_DEBUG("Performing general intersection");
    Supervised::perform_general_intersection<TPB_X, value_idx, value_t>(
      handle, inputs.y, &fss_graph, &ci_graph, params, stream);
  }

  /**
   * Remove zeros
   */
  raft::sparse::op::coo_sort<value_t>(&ci_graph, stream);
  raft::sparse::op::coo_remove_zeros<value_t>(&ci_graph, graph, stream);
  raft::common::nvtx::pop_range();
}

template <typename value_idx, typename value_t, typename umap_inputs, int TPB_X>
void _refine(const raft::handle_t& handle,
             const umap_inputs& inputs,
             UMAPParams* params,
             raft::sparse::COO<value_t>* graph,
             value_t* embeddings)
{
  cudaStream_t stream = handle.get_stream();
  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<TPB_X, value_t>(inputs.n, inputs.d, graph, params, embeddings, stream);
}

template <typename value_idx, typename value_t, typename umap_inputs, int TPB_X>
void _fit(const raft::handle_t& handle,
          const umap_inputs& inputs,
          UMAPParams* params,
          value_t* embeddings,
          raft::sparse::COO<float, int>* graph)
{
  raft::common::nvtx::range fun_scope("umap::unsupervised::fit");

  cudaStream_t stream = handle.get_stream();
  ML::Logger::get().setLevel(params->verbosity);

  UMAPAlgo::_get_graph<value_idx, value_t, umap_inputs, TPB_X>(handle, inputs, params, graph);

  /**
   * Run initialization method
   */
  raft::common::nvtx::push_range("umap::embedding");
  InitEmbed::run(handle, inputs.n, inputs.d, graph, params, embeddings, stream, params->init);

  if (params->callback) {
    params->callback->setup<value_t>(inputs.n, params->n_components);
    params->callback->on_preprocess_end(embeddings);
  }

  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<TPB_X, value_t>(inputs.n, inputs.d, graph, params, embeddings, stream);
  raft::common::nvtx::pop_range();

  if (params->callback) params->callback->on_train_end(embeddings);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename value_idx, typename value_t, typename umap_inputs, int TPB_X>
void _fit_supervised(const raft::handle_t& handle,
                     const umap_inputs& inputs,
                     UMAPParams* params,
                     value_t* embeddings,
                     raft::sparse::COO<float, int>* graph)
{
  raft::common::nvtx::range fun_scope("umap::supervised::fit");

  cudaStream_t stream = handle.get_stream();
  ML::Logger::get().setLevel(params->verbosity);

  UMAPAlgo::_get_graph_supervised<value_idx, value_t, umap_inputs, TPB_X>(
    handle, inputs, params, graph);

  /**
   * Initialize embeddings
   */
  raft::common::nvtx::push_range("umap::supervised::fit");
  InitEmbed::run(handle, inputs.n, inputs.d, graph, params, embeddings, stream, params->init);

  if (params->callback) {
    params->callback->setup<value_t>(inputs.n, params->n_components);
    params->callback->on_preprocess_end(embeddings);
  }

  /**
   * Run simplicial set embedding to approximate low-dimensional representation
   */
  SimplSetEmbed::run<TPB_X, value_t>(inputs.n, inputs.d, graph, params, embeddings, stream);
  raft::common::nvtx::pop_range();

  if (params->callback) params->callback->on_train_end(embeddings);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 *
 */
template <typename value_idx, typename value_t, typename umap_inputs, int TPB_X>
void _transform(const raft::handle_t& handle,
                const umap_inputs& inputs,
                umap_inputs& orig_x_inputs,
                value_t* embedding,
                int embedding_n,
                UMAPParams* params,
                value_t* transformed)
{
  raft::common::nvtx::range fun_scope("umap::transform");
  cudaStream_t stream = handle.get_stream();

  ML::Logger::get().setLevel(params->verbosity);

  CUML_LOG_DEBUG("Running transform");

  CUML_LOG_DEBUG("Building KNN Graph");

  raft::common::nvtx::push_range("umap::knnGraph");
  std::unique_ptr<rmm::device_uvector<value_idx>> knn_indices_b = nullptr;
  std::unique_ptr<rmm::device_uvector<value_t>> knn_dists_b     = nullptr;

  int k = params->n_neighbors;

  knn_graph<value_idx, value_t> knn_graph(inputs.n, k);

  /**
   * If not given precomputed knn graph, compute it
   */

  if (inputs.alloc_knn_graph()) {
    /**
     * Allocate workspace for kNN graph
     */
    knn_indices_b = std::make_unique<rmm::device_uvector<value_idx>>(inputs.n * k, stream);
    knn_dists_b   = std::make_unique<rmm::device_uvector<value_t>>(inputs.n * k, stream);

    knn_graph.knn_indices = knn_indices_b->data();
    knn_graph.knn_dists   = knn_dists_b->data();
  }

  kNNGraph::run<value_idx, value_t, umap_inputs>(
    handle, orig_x_inputs, inputs, knn_graph, k, params, stream);

  raft::common::nvtx::pop_range();

  raft::common::nvtx::push_range("umap::smooth_knn");
  float adjusted_local_connectivity = max(0.0, params->local_connectivity - 1.0);

  CUML_LOG_DEBUG("Smoothing KNN distances");

  /**
   * Perform smooth_knn_dist
   */
  rmm::device_uvector<value_t> sigmas(inputs.n, stream);
  rmm::device_uvector<value_t> rhos(inputs.n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(sigmas.data(), 0, inputs.n * sizeof(value_t), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(rhos.data(), 0, inputs.n * sizeof(value_t), stream));

  dim3 grid_n(raft::ceildiv(inputs.n, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  FuzzySimplSetImpl::smooth_knn_dist<TPB_X, value_idx, value_t>(inputs.n,
                                                                knn_graph.knn_indices,
                                                                knn_graph.knn_dists,
                                                                rhos.data(),
                                                                sigmas.data(),
                                                                params,
                                                                params->n_neighbors,
                                                                adjusted_local_connectivity,
                                                                stream);
  raft::common::nvtx::pop_range();

  /**
   * Compute graph of membership strengths
   */

  int nnz = inputs.n * params->n_neighbors;

  dim3 grid_nnz(raft::ceildiv(nnz, TPB_X), 1, 1);

  CUML_LOG_DEBUG("Executing fuzzy simplicial set");

  /**
   * Allocate workspace for fuzzy simplicial set.
   */

  raft::sparse::COO<value_t> graph_coo(stream, nnz, inputs.n, inputs.n);

  FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X>
    <<<grid_nnz, blk, 0, stream>>>(knn_graph.knn_indices,
                                   knn_graph.knn_dists,
                                   sigmas.data(),
                                   rhos.data(),
                                   graph_coo.vals(),
                                   graph_coo.rows(),
                                   graph_coo.cols(),
                                   graph_coo.n_rows,
                                   params->n_neighbors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  rmm::device_uvector<int> row_ind(inputs.n, stream);
  rmm::device_uvector<int> ia(inputs.n, stream);

  raft::sparse::convert::sorted_coo_to_csr(&graph_coo, row_ind.data(), stream);
  raft::sparse::linalg::coo_degree(&graph_coo, ia.data(), stream);

  rmm::device_uvector<value_t> vals_normed(graph_coo.nnz, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(vals_normed.data(), 0, graph_coo.nnz * sizeof(value_t), stream));

  CUML_LOG_DEBUG("Performing L1 normalization");

  raft::sparse::linalg::csr_row_normalize_l1<value_t>(
    row_ind.data(), graph_coo.vals(), graph_coo.nnz, graph_coo.n_rows, vals_normed.data(), stream);

  init_transform<TPB_X, value_t><<<grid_n, blk, 0, stream>>>(graph_coo.cols(),
                                                             vals_normed.data(),
                                                             graph_coo.n_rows,
                                                             embedding,
                                                             embedding_n,
                                                             params->n_components,
                                                             transformed,
                                                             params->n_neighbors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaMemsetAsync(ia.data(), 0.0, ia.size() * sizeof(int), stream));

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /**
   * Go through raft::sparse::COO values and set everything that's less than
   * vals.max() / params->n_epochs to 0.0
   */
  thrust::device_ptr<value_t> d_ptr = thrust::device_pointer_cast(graph_coo.vals());
  value_t max = *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + nnz));

  int n_epochs = params->n_epochs;
  if (n_epochs <= 0) {
    if (inputs.n <= 10000)
      n_epochs = 100;
    else
      n_epochs = 30;
  } else {
    n_epochs /= 3;
  }

  CUML_LOG_DEBUG("n_epochs=%d", n_epochs);

  raft::linalg::unaryOp<value_t>(
    graph_coo.vals(),
    graph_coo.vals(),
    graph_coo.nnz,
    [=] __device__(value_t input) {
      if (input < (max / float(n_epochs)))
        return 0.0f;
      else
        return input;
    },
    stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  /**
   * Remove zeros
   */
  raft::sparse::COO<value_t> comp_coo(stream);
  raft::sparse::op::coo_remove_zeros<value_t>(&graph_coo, &comp_coo, stream);

  raft::common::nvtx::push_range("umap::optimization");
  CUML_LOG_DEBUG("Computing # of epochs for training each sample");

  rmm::device_uvector<value_t> epochs_per_sample(nnz, stream);

  SimplSetEmbedImpl::make_epochs_per_sample(
    comp_coo.vals(), comp_coo.nnz, n_epochs, epochs_per_sample.data(), stream);

  CUML_LOG_DEBUG("Performing optimization");

  if (params->callback) {
    params->callback->setup<value_t>(inputs.n, params->n_components);
    params->callback->on_preprocess_end(transformed);
  }

  auto initial_alpha = params->initial_alpha / 4.0;

  SimplSetEmbedImpl::optimize_layout<TPB_X, value_t>(transformed,
                                                     inputs.n,
                                                     embedding,
                                                     embedding_n,
                                                     comp_coo.rows(),
                                                     comp_coo.cols(),
                                                     comp_coo.nnz,
                                                     epochs_per_sample.data(),
                                                     params->repulsion_strength,
                                                     params,
                                                     n_epochs,
                                                     stream);
  raft::common::nvtx::pop_range();

  if (params->callback) params->callback->on_train_end(transformed);
}

}  // namespace UMAPAlgo
