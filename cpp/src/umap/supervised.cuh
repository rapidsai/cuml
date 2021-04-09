/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn.hpp>
#include "optimize.cuh"

#include <raft/cudart_utils.h>
#include "fuzzy_simpl_set/runner.cuh"
#include "init_embed/runner.cuh"
#include "knn_graph/runner.cuh"
#include "simpl_set_embed/runner.cuh"

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/linalg/add.cuh>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/filter.cuh>

#include <raft/cuda_utils.cuh>

#include <cuda_runtime.h>

namespace UMAPAlgo {

namespace Supervised {

using namespace ML;

template <int TPB_X, typename T>
__global__ void fast_intersection_kernel(int *rows, int *cols, T *vals, int nnz,
                                         T *target, float unknown_dist = 1.0,
                                         float far_dist = 5.0) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz) {
    int i = rows[row];
    int j = cols[row];
    if (target[i] == T(-1.0) || target[j] == T(-1.0))
      vals[row] *= exp(-unknown_dist);
    else if (target[i] != target[j])
      vals[row] *= exp(-far_dist);
  }
}

template <typename T, int TPB_X>
void reset_local_connectivity(raft::sparse::COO<T> *in_coo,
                              raft::sparse::COO<T> *out_coo,
                              std::shared_ptr<deviceAllocator> d_alloc,
                              cudaStream_t stream  // size = nnz*2
) {
  MLCommon::device_buffer<int> row_ind(d_alloc, stream, in_coo->n_rows);

  raft::sparse::convert::sorted_coo_to_csr(in_coo, row_ind.data(), d_alloc,
                                           stream);

  // Perform l_inf normalization
  raft::sparse::linalg::csr_row_normalize_max<TPB_X, T>(
    row_ind.data(), in_coo->vals(), in_coo->nnz, in_coo->n_rows, in_coo->vals(),
    stream);
  CUDA_CHECK(cudaPeekAtLastError());

  raft::sparse::linalg::coo_symmetrize<TPB_X, T>(
    in_coo, out_coo,
    [] __device__(int row, int col, T result, T transpose) {
      T prod_matrix = result * transpose;
      return result + transpose - prod_matrix;
    },
    d_alloc, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * Combine a fuzzy simplicial set with another fuzzy simplicial set
 * generated from categorical data using categorical distances. The target
 * data is assumed to be categorical label data (a vector of labels),
 * and this will update the fuzzy simplicial set to respect that label
 * data.
 */
template <typename value_t, int TPB_X>
void categorical_simplicial_set_intersection(
  raft::sparse::COO<value_t> *graph_coo, value_t *target, cudaStream_t stream,
  float far_dist = 5.0, float unknown_dist = 1.0) {
  dim3 grid(raft::ceildiv(graph_coo->nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  fast_intersection_kernel<TPB_X, value_t><<<grid, blk, 0, stream>>>(
    graph_coo->rows(), graph_coo->cols(), graph_coo->vals(), graph_coo->nnz,
    target, unknown_dist, far_dist);
}

template <typename value_t, int TPB_X>
__global__ void sset_intersection_kernel(
  int *row_ind1, int *cols1, value_t *vals1, int nnz1, int *row_ind2,
  int *cols2, value_t *vals2, int nnz2, int *result_ind, int *result_cols,
  value_t *result_vals, int nnz, value_t left_min, value_t right_min, int m,
  float mix_weight = 0.5) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int start_idx_res = result_ind[row];
    int stop_idx_res = raft::sparse::get_stop_idx(row, m, nnz, result_ind);

    int start_idx1 = row_ind1[row];
    int stop_idx1 = raft::sparse::get_stop_idx(row, m, nnz1, row_ind1);

    int start_idx2 = row_ind2[row];
    int stop_idx2 = raft::sparse::get_stop_idx(row, m, nnz2, row_ind2);

    for (int j = start_idx_res; j < stop_idx_res; j++) {
      int col = result_cols[j];

      value_t left_val = left_min;
      for (int k = start_idx1; k < stop_idx1; k++) {
        if (cols1[k] == col) {
          left_val = vals1[k];
        }
      }

      value_t right_val = right_min;
      for (int k = start_idx2; k < stop_idx2; k++) {
        if (cols2[k] == col) {
          right_val = vals2[k];
        }
      }

      if (left_val > left_min || right_val > right_min) {
        if (mix_weight < 0.5) {
          result_vals[j] =
            left_val * powf(right_val, mix_weight / (1.0 - mix_weight));
        } else {
          result_vals[j] =
            powf(left_val, (1.0 - mix_weight) / mix_weight) * right_val;
        }
      }
    }
  }
}

/**
 * Computes the CSR column index pointer and values
 * for the general simplicial set intersecftion.
 */
template <typename T, int TPB_X>
void general_simplicial_set_intersection(
  int *row1_ind, raft::sparse::COO<T> *in1, int *row2_ind,
  raft::sparse::COO<T> *in2, raft::sparse::COO<T> *result, float weight,
  std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream) {
  MLCommon::device_buffer<int> result_ind(d_alloc, stream, in1->n_rows);
  CUDA_CHECK(
    cudaMemsetAsync(result_ind.data(), 0, in1->n_rows * sizeof(int), stream));

  int result_nnz = raft::sparse::linalg::csr_add_calc_inds<float, 32>(
    row1_ind, in1->cols(), in1->vals(), in1->nnz, row2_ind, in2->cols(),
    in2->vals(), in2->nnz, in1->n_rows, result_ind.data(), d_alloc, stream);

  result->allocate(result_nnz, in1->n_rows, in1->n_cols, true, stream);

  /**
   * Element-wise sum of two simplicial sets
   */
  raft::sparse::linalg::csr_add_finalize<float, 32>(
    row1_ind, in1->cols(), in1->vals(), in1->nnz, row2_ind, in2->cols(),
    in2->vals(), in2->nnz, in1->n_rows, result_ind.data(), result->cols(),
    result->vals(), stream);

  //@todo: Write a wrapper function for this
  raft::sparse::convert::csr_to_coo<int, TPB_X>(
    result_ind.data(), result->n_rows, result->rows(), result->nnz, stream);

  thrust::device_ptr<const T> d_ptr1 = thrust::device_pointer_cast(in1->vals());
  T min1 = *(thrust::min_element(thrust::cuda::par.on(stream), d_ptr1,
                                 d_ptr1 + in1->nnz));

  thrust::device_ptr<const T> d_ptr2 = thrust::device_pointer_cast(in2->vals());
  T min2 = *(thrust::min_element(thrust::cuda::par.on(stream), d_ptr2,
                                 d_ptr2 + in2->nnz));

  T left_min = max(min1 / 2.0, 1e-8);
  T right_min = max(min2 / 2.0, 1e-8);

  dim3 grid(raft::ceildiv(in1->nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  sset_intersection_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    row1_ind, in1->cols(), in1->vals(), in1->nnz, row2_ind, in2->cols(),
    in2->vals(), in2->nnz, result_ind.data(), result->cols(), result->vals(),
    result->nnz, left_min, right_min, in1->n_rows, weight);
  CUDA_CHECK(cudaGetLastError());

  dim3 grid_n(raft::ceildiv(result->nnz, TPB_X), 1, 1);
}

template <int TPB_X, typename T>
void perform_categorical_intersection(T *y, raft::sparse::COO<T> *rgraph_coo,
                                      raft::sparse::COO<T> *final_coo,
                                      UMAPParams *params,
                                      std::shared_ptr<deviceAllocator> d_alloc,
                                      cudaStream_t stream) {
  float far_dist = 1.0e12;  // target weight
  if (params->target_weights < 1.0)
    far_dist = 2.5 * (1.0 / (1.0 - params->target_weights));

  categorical_simplicial_set_intersection<T, TPB_X>(rgraph_coo, y, stream,
                                                    far_dist);

  raft::sparse::COO<T> comp_coo(d_alloc, stream);
  raft::sparse::op::coo_remove_zeros<TPB_X, T>(rgraph_coo, &comp_coo, d_alloc,
                                               stream);

  reset_local_connectivity<T, TPB_X>(&comp_coo, final_coo, d_alloc, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}

template <int TPB_X, typename value_idx, typename value_t>
void perform_general_intersection(const raft::handle_t &handle, value_t *y,
                                  raft::sparse::COO<value_t> *rgraph_coo,
                                  raft::sparse::COO<value_t> *final_coo,
                                  UMAPParams *params, cudaStream_t stream) {
  auto d_alloc = handle.get_device_allocator();

  /**
   * Calculate kNN for Y
   */
  int knn_dims = rgraph_coo->n_rows * params->target_n_neighbors;
  MLCommon::device_buffer<value_idx> y_knn_indices(d_alloc, stream, knn_dims);
  MLCommon::device_buffer<value_t> y_knn_dists(d_alloc, stream, knn_dims);

  knn_graph<value_idx, value_t> knn_graph(rgraph_coo->n_rows,
                                          params->target_n_neighbors);
  knn_graph.knn_indices = y_knn_indices.data();
  knn_graph.knn_dists = y_knn_dists.data();

  manifold_dense_inputs_t<value_t> y_inputs(y, nullptr, rgraph_coo->n_rows, 1);
  kNNGraph::run<value_idx, value_t, manifold_dense_inputs_t<value_t>>(
    handle, y_inputs, y_inputs, knn_graph, params->target_n_neighbors, params,
    d_alloc, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {
    CUML_LOG_DEBUG("Target kNN Graph");
    std::stringstream ss1, ss2;
    ss1 << raft::arr2Str(y_knn_indices.data(),
                         rgraph_coo->n_rows * params->target_n_neighbors,
                         "knn_indices", stream);
    CUML_LOG_DEBUG("%s", ss1.str().c_str());
    ss2 << raft::arr2Str(y_knn_dists.data(),
                         rgraph_coo->n_rows * params->target_n_neighbors,
                         "knn_dists", stream);
    CUML_LOG_DEBUG("%s", ss2.str().c_str());
  }

  /**
   * Compute fuzzy simplicial set
   */
  raft::sparse::COO<value_t> ygraph_coo(d_alloc, stream);

  FuzzySimplSet::run<TPB_X, value_idx, value_t>(
    rgraph_coo->n_rows, y_knn_indices.data(), y_knn_dists.data(),
    params->target_n_neighbors, &ygraph_coo, params, d_alloc, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {
    CUML_LOG_DEBUG("Target Fuzzy Simplicial Set");
    std::stringstream ss;
    ss << ygraph_coo;
    CUML_LOG_DEBUG(ss.str().c_str());
  }

  /**
   * Compute general simplicial set intersection.
   */
  MLCommon::device_buffer<int> xrow_ind(d_alloc, stream, rgraph_coo->n_rows);
  MLCommon::device_buffer<int> yrow_ind(d_alloc, stream, ygraph_coo.n_rows);

  CUDA_CHECK(cudaMemsetAsync(xrow_ind.data(), 0,
                             rgraph_coo->n_rows * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(yrow_ind.data(), 0,
                             ygraph_coo.n_rows * sizeof(int), stream));

  raft::sparse::COO<value_t> cygraph_coo(d_alloc, stream);
  raft::sparse::op::coo_remove_zeros<TPB_X, value_t>(&ygraph_coo, &cygraph_coo,
                                                     d_alloc, stream);

  raft::sparse::convert::sorted_coo_to_csr(&cygraph_coo, yrow_ind.data(),
                                           d_alloc, stream);
  raft::sparse::convert::sorted_coo_to_csr(rgraph_coo, xrow_ind.data(), d_alloc,
                                           stream);

  raft::sparse::COO<value_t> result_coo(d_alloc, stream);
  general_simplicial_set_intersection<value_t, TPB_X>(
    xrow_ind.data(), rgraph_coo, yrow_ind.data(), &cygraph_coo, &result_coo,
    params->target_weights, d_alloc, stream);

  /**
   * Remove zeros
   */
  raft::sparse::COO<value_t> out(d_alloc, stream);
  raft::sparse::op::coo_remove_zeros<TPB_X, value_t>(&result_coo, &out, d_alloc,
                                                     stream);

  reset_local_connectivity<value_t, TPB_X>(&out, final_coo, d_alloc, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace Supervised
}  // namespace UMAPAlgo
