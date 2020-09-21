/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <matrix/matrix.cuh>
#include <matrix/reverse.cuh>

#include <linalg/unary_op.cuh>

#include <selection/knn.cuh>
#include <sparse/coo.cuh>
#include <sparse/csr.cuh>
#include <sparse/distance.cuh>
#include <sparse/selection.cuh>

#include <common/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>

#include <cuda_utils.cuh>

#include <raft/sparse/cusparse_wrappers.h>

#include <cusparse_v2.h>

#include <sparse/utils.h>

#pragma once

namespace MLCommon {
namespace Sparse {
namespace Selection {

template <typename value_idx, typename value_t>
struct csr_batcher_t {
  csr_batcher_t(value_idx batch_size, value_idx n_rows,
                const value_idx *csr_indptr, const value_idx *csr_indices,
                const value_t *csr_data)
    : batch_start_(0),
      batch_stop_(0),
      batch_rows_(0),
      total_rows_(n_rows),
      batch_size_(batch_size),
      csr_indptr_(csr_indptr),
      csr_indices_(csr_indices),
      csr_data_(csr_data),
      batch_csr_start_offset_(0),
      batch_csr_stop_offset_(0) {}

  void set_batch(int batch_num) {
    batch_start_ = batch_num * batch_size_;
    batch_stop_ = batch_start_ + batch_size_ - 1;  // zero-based indexing

    if (batch_stop_ >= total_rows_)
      batch_stop_ = total_rows_ - 1;  // zero-based indexing

    batch_rows_ = (batch_stop_ - batch_start_) + 1;

    CUML_LOG_DEBUG(
      "Setting batch. batch_start=%d, batch_stop=%d, batch_rows=%d",
      batch_start_, batch_stop_, batch_rows_);
  }

  value_idx get_batch_csr_indptr_nnz(value_idx *batch_indptr,
                                     cudaStream_t stream) {
    MLCommon::Sparse::csr_row_slice_indptr(
      batch_start_, batch_stop_, csr_indptr_, batch_indptr,
      &batch_csr_start_offset_, &batch_csr_stop_offset_, stream);

    CUML_LOG_DEBUG("Computed batch offsets. stop_offset=%d, start_offset=%d",
                   batch_csr_stop_offset_, batch_csr_start_offset_);

    return batch_csr_stop_offset_ - batch_csr_start_offset_;
  }

  void get_batch_csr_indices_data(value_idx *csr_indices, value_t *csr_data,
                                  cudaStream_t stream) {
    MLCommon::Sparse::csr_row_slice_populate(
      batch_csr_start_offset_, batch_csr_stop_offset_, csr_indices_, csr_data_,
      csr_indices, csr_data, stream);
  }

  value_idx batch_rows() const { return batch_rows_; }

  value_idx batch_start() const { return batch_start_; }

  value_idx batch_stop() const { return batch_stop_; }

 private:
  value_idx batch_size_;
  value_idx batch_start_;
  value_idx batch_stop_;
  value_idx batch_rows_;

  value_idx total_rows_;

  const value_idx *csr_indptr_;
  const value_idx *csr_indices_;
  const value_t *csr_data_;

  value_idx batch_csr_start_offset_;
  value_idx batch_csr_stop_offset_;
};

template <typename value_idx>
__global__ void iota_fill_warp_kernel(value_idx *indices, value_idx ncols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  for (int i = tid; i < ncols; i += blockDim.x) {
    indices[row * ncols + i] = i;
  }
}

template <typename value_idx>
void iota_fill(value_idx *indices, value_idx nrows, value_idx ncols,
               cudaStream_t stream) {
  int blockdim = block_dim(ncols);

  iota_fill_warp_kernel<<<nrows, blockdim, 0, stream>>>(indices, ncols);
}

/**
   * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
   * using some distance implementation
   * @param[in] idxIndptr csr indptr of the index matrix (size n_idx_rows + 1)
   * @param[in] idxIndices csr column indices array of the index matrix (size n_idx_nnz)
   * @param[in] idxData csr data array of the index matrix (size idxNNZ)
   * @param[in] idxNNA number of non-zeros for sparse index matrix
   * @param[in] n_idx_rows number of data samples in index matrix
   * @param[in] queryIndptr csr indptr of the query matrix (size n_query_rows + 1)
   * @param[in] queryIndices csr indices array of the query matrix (size queryNNZ)
   * @param[in] queryData csr data array of the query matrix (size queryNNZ)
   * @param[in] queryNNZ number of non-zeros for sparse query matrix
   * @param[in] n_query_rows number of data samples in query matrix
   * @param[in] n_query_cols number of features in query matrix
   * @param[out] output_indices dense matrix for output indices (size n_query_rows * k)
   * @param[out] output_dists dense matrix for output distances (size n_query_rows * k)
   * @param[in] k the number of neighbors to query
   * @param[in] cusparseHandle the initialized cusparseHandle instance to use
   * @param[in] allocator device allocator instance to use
   * @param[in] stream CUDA stream to order operations with respect to
   * @param[in] batch_size_index maximum number of rows to use from index matrix per batch
   * @param[in] batch_size_query maximum number of rows to use from query matrix per batch
   * @param[in] metric distance metric/measure to use
   * @param[in] metricArg potential argument for metric (currently unused)
   * @param[in] expanded_form whether or not Lp variants should be reduced by the pth-root
   */
template <typename value_idx = int, typename value_t = float, int TPB_X = 32>
void brute_force_knn(
  const value_idx *idxIndptr, const value_idx *idxIndices,
  const value_t *idxData, size_t idxNNZ, int n_idx_rows,
  int n_idx_cols, const value_idx *queryIndptr,
  const value_idx *queryIndices, const value_t *queryData, size_t queryNNZ,
  int n_query_rows, int n_query_cols, value_idx *output_indices,
  value_t *output_dists, int k, cusparseHandle_t cusparseHandle,
  std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream,
  size_t batch_size_index = 2 << 14,  // approx 1M
  size_t batch_size_query = 2 << 14,
  ML::MetricType metric = ML::MetricType::METRIC_L2, float metricArg = 0,
  bool expanded_form = false) {
  using namespace raft::sparse;

  bool ascending = true;
  if (metric == ML::MetricType::METRIC_INNER_PRODUCT) ascending = false;

  ML::Distance::DistanceType pw_metric;
  switch(metric) {
    case ML::MetricType::METRIC_INNER_PRODUCT:
      pw_metric = ML::Distance::DistanceType::InnerProduct;
      break;
    case ML::MetricType::METRIC_L2:
      pw_metric = ML::Distance::DistanceType::EucExpandedL2;
      break;
    default:
      THROW("MetricType not supported: %d", metric);
  }

  CUML_LOG_DEBUG("n_query_rows=%d, n_idx_rows=%d", n_query_rows, n_idx_rows);
  int n_batches_query = ceildiv((size_t)n_query_rows, batch_size_query);
  csr_batcher_t<value_idx, value_t> query_batcher(
    batch_size_query, n_query_rows, queryIndptr, queryIndices, queryData);

  size_t rows_processed = 0;

  for (int i = 0; i < n_batches_query; i++) {
    /**
      * Compute index batch info
      */
    CUML_LOG_DEBUG("Beginning index batch %d", i);
    query_batcher.set_batch(i);

    /**
      * Slice CSR to rows in batch
      */
    CUML_LOG_DEBUG("Slicing query CSR for batch. rows=%d out of %d",
                   query_batcher.batch_rows(), n_query_rows);

    device_buffer<value_idx> query_batch_indptr(allocator, stream,
                                                query_batcher.batch_rows() + 1);

    value_idx n_query_batch_nnz =
      query_batcher.get_batch_csr_indptr_nnz(query_batch_indptr.data(), stream);

    device_buffer<value_idx> query_batch_indices(allocator, stream,
                                                 n_query_batch_nnz);
    device_buffer<value_t> query_batch_data(allocator, stream,
                                            n_query_batch_nnz);

    query_batcher.get_batch_csr_indices_data(query_batch_indices.data(),
                                             query_batch_data.data(), stream);

    // A 3-partition temporary merge space to scale the batching. 2 parts for subsequent
    // batches and 1 space for the results of the merge, which get copied back to the top
    device_buffer<value_idx> merge_buffer_indices(allocator, stream, 0);
    device_buffer<value_t> merge_buffer_dists(allocator, stream, 0);

    value_t *dists_merge_buffer_ptr;
    value_idx *indices_merge_buffer_ptr;

    int n_batches_idx = ceildiv((size_t)n_idx_rows, batch_size_index);
    csr_batcher_t<value_idx, value_t> idx_batcher(
      batch_size_index, n_idx_rows, idxIndptr, idxIndices, idxData);

    for (int j = 0; j < n_batches_idx; j++) {
      CUML_LOG_DEBUG("Beginning query batch %d", j);
      idx_batcher.set_batch(j);

      merge_buffer_indices.resize(query_batcher.batch_rows() * k * 3, stream);
      merge_buffer_dists.resize(query_batcher.batch_rows() * k * 3, stream);

      /**
        * Slice CSR to rows in batch
      */
      CUML_LOG_DEBUG("Slicing index CSR for batch. rows=%d out of %d",
                     idx_batcher.batch_rows(), n_idx_rows);
      device_buffer<value_idx> idx_batch_indptr(allocator, stream,
                                                idx_batcher.batch_rows() + 1);
      device_buffer<value_idx> idx_batch_indices(allocator, stream, 0);
      device_buffer<value_t> idx_batch_data(allocator, stream, 0);

      value_idx idx_batch_nnz =
        idx_batcher.get_batch_csr_indptr_nnz(idx_batch_indptr.data(), stream);

      idx_batch_indices.resize(idx_batch_nnz, stream);
      idx_batch_data.resize(idx_batch_nnz, stream);

      idx_batcher.get_batch_csr_indices_data(idx_batch_indices.data(),
                                             idx_batch_data.data(), stream);

      /**
       * Compute distances
       */
      CUML_LOG_DEBUG("Computing pairwise distances for batch");
      MLCommon::Sparse::Distance::distances_config_t<value_idx, value_t>
        dist_config;
      dist_config.b_nrows = idx_batcher.batch_rows();
      dist_config.b_ncols = n_idx_cols;
      dist_config.b_nnz = idx_batch_nnz;

      dist_config.b_indptr = idx_batch_indptr.data();
      dist_config.b_indices = idx_batch_indices.data();
      dist_config.b_data = idx_batch_data.data();

      dist_config.a_nrows = query_batcher.batch_rows();
      dist_config.a_ncols = n_query_cols;
      dist_config.a_nnz = n_query_batch_nnz;

      dist_config.a_indptr = query_batch_indptr.data();
      dist_config.a_indices = query_batch_indices.data();
      dist_config.a_data = query_batch_data.data();

      dist_config.handle = cusparseHandle;
      dist_config.allocator = allocator;
      dist_config.stream = stream;

      value_idx dense_size =
        idx_batcher.batch_rows() * query_batcher.batch_rows();
      device_buffer<value_t> batch_dists(allocator, stream, dense_size);


      Distance::pairwiseDistance(batch_dists.data(), dist_config, pw_metric);

      idx_batch_indptr.release(stream);
      idx_batch_indices.release(stream);
      idx_batch_data.release(stream);

      // Build batch indices array
      device_buffer<value_idx> batch_indices(allocator, stream,
                                             batch_dists.size());

      // populate batch indices array
      value_idx batch_rows = query_batcher.batch_rows(),
                batch_cols = idx_batcher.batch_rows();

      iota_fill(batch_indices.data(), batch_rows, batch_cols, stream);

      /**
       * Perform k-selection on batch & merge with other k-selections
       */
      CUML_LOG_DEBUG("Performing k-selection");
      size_t merge_buffer_offset = batch_rows * k;
      dists_merge_buffer_ptr = merge_buffer_dists.data() + merge_buffer_offset;
      indices_merge_buffer_ptr =
        merge_buffer_indices.data() + merge_buffer_offset;

      // build translation buffer to shift resulting indices by the batch
      std::vector<value_idx> id_ranges;
      id_ranges.push_back(0);
      id_ranges.push_back(idx_batcher.batch_start());

      // in the case where the number of idx rows in the batch is < k, we
      // want to adjust k.
      value_idx n_neighbors = min(k, batch_cols);

      // kernel to slice first (min) k cols and copy into batched merge buffer
      select_k(batch_dists.data(), batch_indices.data(), batch_rows, batch_cols,
               dists_merge_buffer_ptr, indices_merge_buffer_ptr, ascending,
               n_neighbors, stream);

      // Perform necessary post-processing
      if ((metric == ML::MetricType::METRIC_L2 ||
           metric == ML::MetricType::METRIC_Lp) &&
          !expanded_form) {
        /**
          * post-processing
          */
        value_t p = 0.5;  // standard l2
        if (metric == ML::MetricType::METRIC_Lp) p = 1.0 / metricArg;
        MLCommon::LinAlg::unaryOp<value_t>(
          dists_merge_buffer_ptr, dists_merge_buffer_ptr, batch_rows * k,
          [p] __device__(value_t input) {
            int neg = input < 0 ? -1 : 1;
            return powf(fabs(input), p) * neg;
          },
          stream);
      }

      value_t *dists_merge_buffer_tmp_ptr = dists_merge_buffer_ptr;
      value_idx *indices_merge_buffer_tmp_ptr = indices_merge_buffer_ptr;

      // Merge results of difference batches if necessary
      if (idx_batcher.batch_start() > 0) {
        device_buffer<value_idx> trans(allocator, stream, id_ranges.size());
        updateDevice(trans.data(), id_ranges.data(), id_ranges.size(), stream);

        size_t merge_buffer_tmp_out = batch_rows * k * 2;
        dists_merge_buffer_tmp_ptr =
          merge_buffer_dists.data() + merge_buffer_tmp_out;
        indices_merge_buffer_tmp_ptr =
          merge_buffer_indices.data() + merge_buffer_tmp_out;

        CUML_LOG_DEBUG("Running merge parts");

        // combine merge buffers only if there's more than 1 partition to combine
        MLCommon::Selection::knn_merge_parts(
          merge_buffer_dists.data(), merge_buffer_indices.data(),
          dists_merge_buffer_tmp_ptr, indices_merge_buffer_tmp_ptr,
          query_batcher.batch_rows(), 2, k, stream, trans.data());

        CUML_LOG_DEBUG("Done.");
      }

      CUML_LOG_DEBUG("Performing copy async");

      // copy merged output back into merge buffer partition for next iteration
      copyAsync<value_idx>(merge_buffer_indices.data(),
                           indices_merge_buffer_tmp_ptr, batch_rows * k,
                           stream);
      copyAsync<value_t>(merge_buffer_dists.data(), dists_merge_buffer_tmp_ptr,
                         batch_rows * k, stream);

      CUML_LOG_DEBUG("Done.");
    }

    // Copy final merged batch to output array
    copyAsync<value_idx>(output_indices + (rows_processed * k),
                         merge_buffer_indices.data(),
                         query_batcher.batch_rows() * k, stream);
    copyAsync<value_t>(output_dists + (rows_processed * k),
                       merge_buffer_dists.data(),
                       query_batcher.batch_rows() * k, stream);

    rows_processed += query_batcher.batch_rows();
  }
}

};  // END namespace Selection
};  // END namespace Sparse
};  // END namespace MLCommon
