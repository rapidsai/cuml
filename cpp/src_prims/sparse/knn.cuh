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
#include <sparse/distances.cuh>
#include <sparse/selection.cuh>

#include <common/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>

#include <raft/sparse/cusparse_wrappers.h>

#include <cusparse_v2.h>

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
    batch_stop_ = batch_start_ + batch_size_;

    if (batch_stop_ >= total_rows_) {
      batch_stop_ = total_rows_ - 1;
      batch_rows_ = (batch_stop_ - batch_start_) + 1;
    } else {
      batch_rows_ = batch_stop_ - batch_start_;
    }
  }

  value_idx get_batch_csr_indptr_nnz(value_idx *batch_indptr,
                                     cudaStream_t stream) {
    MLCommon::Sparse::csr_row_slice_indptr(
      batch_start_, batch_stop_, csr_indptr_, batch_indptr,
      &batch_csr_start_offset_, &batch_csr_stop_offset_, stream);

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
__global__ void iota_fill_kernel(value_idx *indices, value_idx nrows,
                                 value_idx ncols) {
  value_idx index = blockIdx.x * blockDim.x + threadIdx.x;

  value_idx i = index / ncols, j = index % ncols;

  if (i < nrows && j < ncols) indices[i * ncols + j] = j;
}

/**
   * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
   * using some distance implementation
   * @param allocator the device memory allocator to use for temporary scratch memory
   * @param userStream the main cuda stream to use
   * @param translations translation ids for indices when index rows represent
   *        non-contiguous partitions
   * @param metric corresponds to the FAISS::metricType enum (default is euclidean)
   * @param metricArg metric argument to use. Corresponds to the p arg for lp norm
   * @param expanded_form whether or not lp variants should be reduced w/ lp-root
   */
template <typename value_idx = int, typename value_t = float, int TPB_X = 32>
void brute_force_knn(
  const value_idx *idxIndptr, const value_idx *idxIndices,
  const value_t *idxData, size_t idxNNZ, value_idx n_idx_rows,
  value_idx n_idx_cols, const value_idx *queryIndptr,
  const value_idx *queryIndices, const value_t *queryData, size_t queryNNZ,
  value_idx n_query_rows, value_idx n_query_cols, value_idx *output_indices,
  value_t *output_dists, int k, cusparseHandle_t cusparseHandle,
  std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream,
  size_t batch_size = 2 << 20,  // approx 1M
  ML::MetricType metric = ML::MetricType::METRIC_L2, float metricArg = 0,
  bool expanded_form = false) {
  using namespace raft::sparse;

  int n_batches_query = ceildiv((size_t)n_query_rows, batch_size);
  bool ascending = true;
  if (metric == ML::MetricType::METRIC_INNER_PRODUCT) ascending = false;

  csr_batcher_t<value_idx, value_t> query_batcher(
    batch_size, n_query_rows, queryIndptr, queryIndices, queryData);
  for (int i = 0; i < n_batches_query; i++) {
    query_batcher.set_batch(i);

    /**
	 * Slice CSR to rows in batch
	 */
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
    // batches and 1 space for the results of the merge, which get copied back to the
    device_buffer<value_idx> merge_buffer_indices(
      allocator, stream, k * query_batcher.batch_rows() * 3);
    device_buffer<value_t> merge_buffer_dists(
      allocator, stream, k * query_batcher.batch_rows() * 3);

    value_t *dists_merge_buffer_ptr;
    value_idx *indices_merge_buffer_ptr;

    int n_batches_idx = ceildiv((size_t)n_idx_rows, batch_size);

    for (int j = 0; j < n_batches_idx; j++) {
      /**
        * Compute index batch info
		*/
      csr_batcher_t<value_idx, value_t> idx_batcher(
        batch_size, n_idx_rows, idxIndptr, idxIndices, idxData);
      idx_batcher.set_batch(j);

      /**
   	   * Slice CSR to rows in batch
	   */
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
       * Transpose index array into csc
       */
      device_buffer<value_idx> csc_idx_batch_indptr(allocator, stream,
                                                    n_idx_cols + 1);
      device_buffer<value_idx> csc_idx_batch_indices(allocator, stream,
                                                     idx_batch_nnz);
      device_buffer<value_t> csc_idx_batch_data(allocator, stream,
                                                idx_batch_nnz);

      csr_transpose(cusparseHandle, idx_batch_indptr.data(),
                    idx_batch_indices.data(), idx_batch_data.data(),
                    csc_idx_batch_indptr.data(), csc_idx_batch_indices.data(),
                    csc_idx_batch_data.data(), idx_batcher.batch_rows(),
                    n_idx_cols, idx_batch_nnz, allocator, stream);

      /**
       * Compute distances
       */
      MLCommon::Sparse::Distance::distances_config_t<value_idx, value_t>
        dist_config;
      dist_config.index_nrows = idx_batcher.batch_rows();
      dist_config.index_ncols = n_idx_cols;
      dist_config.index_nnz = idx_batch_nnz;

      dist_config.csc_index_indptr = csc_idx_batch_indptr.data();
      dist_config.csc_index_indices = csc_idx_batch_indices.data();
      dist_config.csc_index_data = csc_idx_batch_data.data();

      dist_config.search_nrows = query_batcher.batch_rows();
      dist_config.search_ncols = n_query_cols;
      dist_config.search_nnz = n_query_batch_nnz;

      dist_config.csr_search_indptr = query_batch_indptr.data();
      dist_config.csr_search_indices = query_batch_indices.data();
      dist_config.csr_search_data = query_batch_data.data();

      dist_config.handle = cusparseHandle;
      dist_config.allocator = allocator;
      dist_config.stream = stream;

      value_idx dense_size =
        idx_batcher.batch_rows() * query_batcher.batch_rows();
      device_buffer<value_t> batch_dists(allocator, stream, dense_size);

      if (metric == ML::MetricType::METRIC_INNER_PRODUCT) {
        Distance::ip_distances_t<value_idx, value_t> compute_dists(dist_config);
        compute_dists.compute(batch_dists.data());
      } else if (metric == ML::MetricType::METRIC_L2) {
        Distance::l2_distances_t<value_idx, value_t> compute_dists(dist_config);
        compute_dists.compute(batch_dists.data());
      } else {
        throw "MetricType not supported";
      }

      idx_batch_indptr.release(stream);
      idx_batch_indices.release(stream);
      idx_batch_data.release(stream);

      csc_idx_batch_indptr.release(stream);
      csc_idx_batch_indices.release(stream);
      csc_idx_batch_data.release(stream);

      // Build batch indices array
      device_buffer<value_idx> batch_indices(allocator, stream,
                                             batch_dists.size());

      // populate batch indices array
      value_idx batch_rows = query_batcher.batch_rows(),
                batch_cols = idx_batcher.batch_rows();
      iota_fill_kernel<<<ceildiv(batch_rows * batch_cols, 256), 256, 0,
                         stream>>>(batch_indices.data(), batch_rows,
                                   batch_cols);

      /**
       * Perform k-selection on batch & merge with other k-selections
       */
      size_t merge_buffer_offset = batch_rows * k;
      dists_merge_buffer_ptr = merge_buffer_dists.data() + merge_buffer_offset;
      indices_merge_buffer_ptr =
        merge_buffer_indices.data() + merge_buffer_offset;

      // build translation buffer to shift resulting indices by the batch
      std::vector<value_idx> id_ranges;
      id_ranges.push_back(0);
      id_ranges.push_back(idx_batcher.batch_start());

      // kernel to slice first (min) k cols and copy into batched merge buffer
      select_k(batch_dists.data(), batch_indices.data(), batch_rows, batch_cols,
               dists_merge_buffer_ptr, indices_merge_buffer_ptr, ascending, k,
               stream,
               /*translation for current batch*/
               id_ranges[1]);

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
          [p] __device__(value_t input) { return powf(input, p); }, stream);
      }

      size_t merge_buffer_tmp_out = batch_rows * k * 2;
      value_t *dists_merge_buffer_tmp_ptr = dists_merge_buffer_ptr;
      value_idx *indices_merge_buffer_tmp_ptr = indices_merge_buffer_ptr;

      // Merge results of difference batches if necessary
      if (idx_batcher.batch_start() > 0) {
        dists_merge_buffer_tmp_ptr =
          merge_buffer_dists.data() + merge_buffer_tmp_out;
        indices_merge_buffer_tmp_ptr =
          merge_buffer_indices.data() + merge_buffer_tmp_out;

        // combine merge buffers only if there's more than 1 partition to combine
        MLCommon::Selection::knn_merge_parts(
          dists_merge_buffer_ptr, indices_merge_buffer_ptr,
          dists_merge_buffer_tmp_ptr, indices_merge_buffer_tmp_ptr,
          query_batcher.batch_rows(), 2, k, stream, id_ranges.data());
      }

      // copy merged output back into merge buffer partition for next iteration
      copyAsync(merge_buffer_indices.data(), indices_merge_buffer_tmp_ptr,
                batch_rows * k, stream);
      copyAsync(merge_buffer_dists.data(), dists_merge_buffer_tmp_ptr,
                batch_rows * k, stream);
    }

    // Copy final merged batch to output array
    copyAsync(output_indices, indices_merge_buffer_ptr,
              query_batcher.batch_rows() * k, stream);
    copyAsync(output_dists, dists_merge_buffer_ptr,
              query_batcher.batch_rows() * k, stream);
  }
}

};  // END namespace Selection
};  // END namespace Sparse
};  // END namespace MLCommon
