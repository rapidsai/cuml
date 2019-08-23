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

#include "optimize.h"
#include "umapparams.h"

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

#include "knn/knn.hpp"

#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <iostream>

namespace UMAPAlgo {

namespace Supervised {

using namespace ML;

using namespace MLCommon::Sparse;

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
void reset_local_connectivity(COO<T> *in_coo, COO<T> *out_coo,
                              cudaStream_t stream  // size = nnz*2
) {
  dim3 grid_n(MLCommon::ceildiv(in_coo->n_rows, TPB_X), 1, 1);
  dim3 blk_n(TPB_X, 1, 1);

  int *row_ind;
  MLCommon::allocate(row_ind, in_coo->n_rows);

  MLCommon::Sparse::sorted_coo_to_csr(in_coo, row_ind, stream);

  // Perform l_inf normalization
  MLCommon::Sparse::csr_row_normalize_max<TPB_X, T>(
    row_ind, in_coo->vals, in_coo->nnz, in_coo->n_rows, in_coo->vals, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::Sparse::coo_symmetrize<TPB_X, T>(
    in_coo, out_coo,
    [] __device__(int row, int col, T result, T transpose) {
      T prod_matrix = result * transpose;
      return result + transpose - prod_matrix;
    },
    stream);

  CUDA_CHECK(cudaFree(row_ind));
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
         * Combine a fuzzy simplicial set with another fuzzy simplicial set
         * generated from categorical data using categorical distances. The target
         * data is assumed to be categorical label data (a vector of labels),
         * and this will update the fuzzy simplicial set to respect that label
         * data.
         */
template <typename T, int TPB_X>
void categorical_simplicial_set_intersection(COO<T> *graph_coo, T *target,
                                             cudaStream_t stream,
                                             float far_dist = 5.0,
                                             float unknown_dist = 1.0) {
  dim3 grid(MLCommon::ceildiv(graph_coo->nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  fast_intersection_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(
    graph_coo->rows, graph_coo->cols, graph_coo->vals, graph_coo->nnz, target,
    unknown_dist, far_dist);
}

template <typename T, int TPB_X>
__global__ void sset_intersection_kernel(int *row_ind1, int *cols1, T *vals1,
                                         int nnz1, int *row_ind2, int *cols2,
                                         T *vals2, int nnz2, int *result_ind,
                                         int *result_cols, T *result_vals,
                                         int nnz, T left_min, T right_min,
                                         int m, float mix_weight = 0.5) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int start_idx_res = result_ind[row];
    int stop_idx_res = MLCommon::Sparse::get_stop_idx(row, m, nnz, result_ind);

    int start_idx1 = row_ind1[row];
    int stop_idx1 = MLCommon::Sparse::get_stop_idx(row, m, nnz1, row_ind1);

    int start_idx2 = row_ind2[row];
    int stop_idx2 = MLCommon::Sparse::get_stop_idx(row, m, nnz2, row_ind2);

    for (int j = start_idx_res; j < stop_idx_res; j++) {
      int col = result_cols[j];

      T left_val = left_min;
      for (int k = start_idx1; k < stop_idx1; k++) {
        if (cols1[k] == col) {
          left_val = vals1[k];
        }
      }

      T right_val = right_min;
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
void general_simplicial_set_intersection(int *row1_ind, COO<T> *in1,
                                         int *row2_ind, COO<T> *in2,
                                         COO<T> *result, float weight,
                                         cudaStream_t stream) {
  int *result_ind;
  MLCommon::allocate(result_ind, in1->n_rows, true);

  int result_nnz = MLCommon::Sparse::csr_add_calc_inds<float, 32>(
    row1_ind, in1->cols, in1->vals, in1->nnz, row2_ind, in2->cols, in2->vals,
    in2->nnz, in1->n_rows, result_ind, stream);

  result->allocate(result_nnz, in1->n_rows);

  /**
             * Element-wise sum of two simplicial sets
             */
  MLCommon::Sparse::csr_add_finalize<float, 32>(
    row1_ind, in1->cols, in1->vals, in1->nnz, row2_ind, in2->cols, in2->vals,
    in2->nnz, in1->n_rows, result_ind, result->cols, result->vals, stream);

  //@todo: Write a wrapper function for this
  MLCommon::Sparse::csr_to_coo<TPB_X>(result_ind, result->n_rows, result->rows,
                                      result->nnz, stream);

  thrust::device_ptr<const T> d_ptr1 = thrust::device_pointer_cast(in1->vals);
  T min1 = *(thrust::min_element(thrust::cuda::par.on(stream), d_ptr1,
                                 d_ptr1 + in1->nnz));

  thrust::device_ptr<const T> d_ptr2 = thrust::device_pointer_cast(in2->vals);
  T min2 = *(thrust::min_element(thrust::cuda::par.on(stream), d_ptr2,
                                 d_ptr2 + in2->nnz));

  T left_min = max(min1 / 2.0, 1e-8);
  T right_min = max(min2 / 2.0, 1e-8);

  dim3 grid(MLCommon::ceildiv(in1->nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  sset_intersection_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    row1_ind, in1->cols, in1->vals, in1->nnz, row2_ind, in2->cols, in2->vals,
    in2->nnz, result_ind, result->cols, result->vals, result->nnz, left_min,
    right_min, in1->n_rows, weight);

  dim3 grid_n(MLCommon::ceildiv(result->nnz, TPB_X), 1, 1);

  CUDA_CHECK(cudaFree(result_ind));
}

template <int TPB_X, typename T>
void perform_categorical_intersection(T *y, COO<T> *rgraph_coo,
                                      COO<T> *final_coo, UMAPParams *params,
                                      cudaStream_t stream) {
  float far_dist = 1.0e12;  // target weight
  if (params->target_weights < 1.0)
    far_dist = 2.5 * (1.0 / (1.0 - params->target_weights));

  categorical_simplicial_set_intersection<T, TPB_X>(rgraph_coo, y, stream,
                                                    far_dist);

  COO<T> comp_coo;
  coo_remove_zeros<TPB_X, T>(rgraph_coo, &comp_coo, stream);

  reset_local_connectivity<T, TPB_X>(&comp_coo, final_coo, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}

template <int TPB_X, typename T>
void perform_general_intersection(const cumlHandle &handle, T *y,
                                  COO<T> *rgraph_coo, COO<T> *final_coo,
                                  UMAPParams *params, cudaStream_t stream) {
  /**
             * Calculate kNN for Y
             */
  long *y_knn_indices;
  T *y_knn_dists;

  int knn_dims = rgraph_coo->n_rows * params->target_n_neighbors;

  MLCommon::allocate(y_knn_indices, knn_dims, true);
  MLCommon::allocate(y_knn_dists, knn_dims, true);

  kNNGraph::run(y, rgraph_coo->n_rows, y, rgraph_coo->n_rows, 1, y_knn_indices,
                y_knn_dists, params->target_n_neighbors, params, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  if (params->verbose) {
    std::cout << "Target kNN Graph" << std::endl;
    std::cout << MLCommon::arr2Str(
                   y_knn_indices,
                   rgraph_coo->n_rows * params->target_n_neighbors,
                   "knn_indices", stream)
              << std::endl;
    std::cout << MLCommon::arr2Str(
                   y_knn_dists, rgraph_coo->n_rows * params->target_n_neighbors,
                   "knn_dists", stream)
              << std::endl;
  }

  /**
             * Compute fuzzy simplicial set
             */
  COO<T> ygraph_coo;
  FuzzySimplSet::run<TPB_X, T>(rgraph_coo->n_rows, y_knn_indices, y_knn_dists,
                               params->target_n_neighbors, &ygraph_coo, params,
                               stream);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaFree(y_knn_indices));
  CUDA_CHECK(cudaFree(y_knn_dists));

  if (params->verbose) {
    std::cout << "Target Fuzzy Simplicial Set" << std::endl;
    std::cout << ygraph_coo << std::endl;
  }

  /**
             * Compute general simplicial set intersection.
             */
  int *xrow_ind, *yrow_ind;
  MLCommon::allocate(xrow_ind, rgraph_coo->n_rows, true);
  MLCommon::allocate(yrow_ind, ygraph_coo.n_rows, true);

  COO<T> cygraph_coo;
  coo_remove_zeros<TPB_X, T>(&ygraph_coo, &cygraph_coo, stream);

  MLCommon::Sparse::sorted_coo_to_csr(&cygraph_coo, yrow_ind, stream);
  MLCommon::Sparse::sorted_coo_to_csr(rgraph_coo, xrow_ind, stream);

  COO<T> result_coo;
  general_simplicial_set_intersection<T, TPB_X>(xrow_ind, rgraph_coo, yrow_ind,
                                                &cygraph_coo, &result_coo,
                                                params->target_weights, stream);

  CUDA_CHECK(cudaFree(xrow_ind));
  CUDA_CHECK(cudaFree(yrow_ind));

  /**
             * Remove zeros
             */
  COO<T> out;
  coo_remove_zeros<TPB_X, T>(&result_coo, &out, stream);

  result_coo.destroy();

  reset_local_connectivity<T, TPB_X>(&out, final_coo, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace Supervised
}  // namespace UMAPAlgo
