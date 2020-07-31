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


#pragma once

#include <matrix/reverse.cuh>
#include <matrix/matrix.cuh>

#include <selection/knn.cuh>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include "cusparse_wrappers.h"
#include <common/device_buffer.hpp>
#include <common/cudart_utils.h>
#include <cuda_utils.cuh>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Selection {


template<typename value_idx>
void csr_row_slice_indptr(value_idx start_row, value_idx stop_row,
		const value_idx *indptr, value_idx *indptr_out,
		value_idx *start_offset, value_idx *stop_offset, cudaStream_t stream) {

	updateHost(start_offset, indptr+start_row, 1, stream);
	updateHost(stop_offset, indptr+stop_row+1, 1, stream);
	CUDA_CHECK(cudaStreamSynchronize(stream));

	copyAsync(indptr_out, indptr+start_row, indptr+stop_row+1, stream);
}

template<typename value_idx, typename value_t>
void csr_row_slice_populate(value_idx start_offset, value_idx stop_offset,
		const value_idx *indices, const value_t *data,
		value_idx *indices_out, value_t *data_out, cudaStream_t stream) {

	copy(indices_out, indices+start_offset, indices+stop_offset, stream);
	copy(data_out, data+start_offset, data+stop_offset, stream);
}

template <typename K,
          typename IndexType,
          int warp_q,
          int thread_q,
          int tpb>
__global__ void select_k_kernel(K *inK,
                        IndexType *inV,
                        size_t n_rows,
                        size_t n_cols,
 					    K *outK,
					    IndexType *outV,
					    K initK,
					    IndexType initV,
					    bool select_min,
					    int k,
					    int64_t translation = 0) {
  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ K smemK[kNumWarps * warp_q];
  __shared__ IndexType smemV[kNumWarps * warp_q];

  faiss::gpu::BlockSelect<K, IndexType, false, faiss::gpu::Comparator<K>, warp_q, thread_q, tpb>
    heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;

  int i = threadIdx.x;
  K* inKStart = inK + (row * k  + i);
  IndexType* inVStart = inV + (row * k  + i);

  // Whole warps must participate in the selection
  int limit = faiss::gpu::utils::roundDown(n_cols, faiss::gpu::kWarpSize);

  for (; i < limit; i += tpb) {
    heap.add(*inKStart, (*inVStart) + translation);
    inKStart += tpb;
    inVStart += tpb;
  }

  // Handle last remainder fraction of a warp of elements
  if (i < n_cols) {
    heap.addThreadQ(*inKStart, (*inVStart) + translation);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k  + i] = smemK[i];
    outV[row * k  + i] = smemV[i];
  }
}

template <int warp_q, int thread_q>
inline void select_k_impl(float *inK,
    					  int64_t *inV,
						  size_t n_rows,
						  size_t n_cols,
						  float *outK,
						  int64_t *outV,
						  bool select_min,
						  int k,
						  cudaStream_t stream,
						  int64_t translation = 0) {

  auto grid = dim3(n_rows);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block = dim3(n_threads);

  auto kInit = select_min ? faiss::gpu::Limits<float>::getMin() : faiss::gpu::Limits<float>::getMax();
  auto vInit = -1;
  select_k_kernel<float, int64_t, warp_q, thread_q, n_threads>
    <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV,
                                 kInit, vInit, k, translation);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param n_samples number of samples per partition
 * @param n_parts number of partitions
 * @param k number of neighbors per partition (also number of merged neighbors)
 * @param stream CUDA stream to use
 * @param translations mapping of index offsets for each partition
 */
inline void select_k(float *inK, int64_t *inV, size_t n_rows, size_t n_cols,
							float *outK, int64_t *outV, bool select_min, int k,
                            cudaStream_t stream, int64_t translation = 0) {
  if (k == 1)
	  select_k_impl<1, 1>(inK, inV,  n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 32)
	  select_k_impl<32, 2>(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 64)
	  select_k_impl<64, 3>(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 128)
	  select_k_impl<128, 3>(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 256)
	  select_k_impl<256, 4>(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 512)
	  select_k_impl<512, 8>(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
  else if (k <= 1024)
	  select_k_impl<1024, 8>(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream, translation);
}



/**
   * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
   * @param allocator the device memory allocator to use for temporary scratch memory
   * @param userStream the main cuda stream to use
   * @param translations translation ids for indices when index rows represent
   *        non-contiguous partitions
   * @param metric corresponds to the FAISS::metricType enum (default is euclidean)
   * @param metricArg metric argument to use. Corresponds to the p arg for lp norm
   * @param expanded_form whether or not lp variants should be reduced w/ lp-root
   */
template <typename value_idx = int64_t, typename value_t = float, int TPB_X=32>
void brute_force_knn(const value_idx *idxIndptr,
					 const value_idx *idxIndices,
					 const value_t *idxData,
					 size_t idxNNZ,
					 size_t n_idx_rows, size_t n_idx_cols,
					 const value_idx *queryIndptr,
					 const value_idx *queryIndices,
					 const value_t *queryData,
					 size_t queryNNZ,
					 size_t n_query_rows, size_t n_query_cols,
					 value_idx *output_indices,
					 value_t *output_dists,
					 int k,
					 cusparseHandle_t cusparseHandle,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t stream,
					 size_t batch_size = 2<<20, // approx 1M
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false) {

	int n_batches_query = ceildiv(n_query_rows, batch_size);
	bool ascending = true;
	if(metric == ML::MetricType::METRIC_INNER_PRODUCT)
		ascending = false;

	value_t alpha = 1.0, beta = 0.0;

	for(int i = 0; i < n_batches_query; i++) {

		// @TODO: This batching logic can likely be refactored into helper functions or
		// some sort of class that can manage the state internally.

		/**
		 * Compute index batch info
		 */
		value_idx query_batch_start = i * batch_size;
		value_idx query_batch_stop = query_batch_start + batch_size;
		value_idx n_query_batch_rows = query_batch_stop - query_batch_start;

		if(query_batch_stop >= n_query_rows)
			query_batch_stop = n_query_rows-1;

		// TODO: When batching is not necessary, just use the input directly instead of copying.

		/**
		 * Slice CSR to rows in batch
		 */
		device_buffer<value_idx> query_batch_indptr(allocator, stream, n_query_batch_rows);

		value_idx query_start_offset, query_stop_offset;

		csr_row_slice_indptr(query_batch_start, query_batch_stop, queryIndptr, query_batch_indptr.data(),
				&query_start_offset, &query_stop_offset, stream);

		value_idx n_query_batch_elms = query_stop_offset - query_start_offset;

		device_buffer<value_idx> query_batch_indices(allocator, stream, n_query_batch_elms);
		device_buffer<value_idx> query_batch_data(allocator, stream, n_query_batch_elms);

		csr_row_slice_populate(query_start_offset, query_stop_offset,
				queryIndptr, queryData, query_batch_indices.data(), query_batch_data.data(), stream);

		/**
		 * Create cusparse descriptors
		 */
		cusparseSpMatDescr_t matA;
		CUSPARSE_CHECK(cusparsecreatecsr(&matA, n_query_rows, n_query_cols, queryNNZ, queryIndptr, queryIndices, queryData));

		// A 3-partition temporary merge space to scale the batching. 2 parts for subsequent
		// batches and 1 space for the results of the merge, which get copied back to the
		//
		device_buffer<value_idx> merge_buffer_indices(allocator, stream, k * n_query_rows * 3);
		device_buffer<value_t> merge_buffer_dists(allocator, stream, k * n_query_rows * 3);

	    value_t *dists_merge_buffer_ptr;
	    value_idx *indices_merge_buffer_ptr;

		int n_batches_idx = ceildiv(n_idx_rows, batch_size);

		for(int j = 0; j < n_batches_idx; j++) {

			/**
			 * Compute query batch info
			 */
			value_idx idx_batch_start = j * batch_size;
			value_idx idx_batch_stop = idx_batch_start + batch_size;
			value_idx n_idx_batch_rows = idx_batch_stop - idx_batch_start;

			if(idx_batch_stop >= n_idx_rows)
				idx_batch_stop = n_idx_rows-1;

			/**
			 * Slice CSR to rows in batch
			 */
			device_buffer<value_idx> idx_batch_indptr(allocator, stream, n_idx_batch_rows);

			value_idx idx_start_offset, idx_stop_offset;

			csr_row_slice_indptr(idx_batch_start, idx_batch_stop, idxIndptr, idx_batch_indptr.data(),
					&idx_start_offset, &idx_stop_offset, stream);

			value_idx n_idx_batch_elms = idx_stop_offset - idx_start_offset;

			device_buffer<value_idx> idx_batch_indices(allocator, stream, n_idx_batch_elms);
			device_buffer<value_idx> idx_batch_data(allocator, stream, n_idx_batch_elms);

			csr_row_slice_populate(idx_start_offset, idx_stop_offset,
					idxIndptr, idxData, idx_batch_indices.data(), idx_batch_data.data(), stream);

			/**
			 * Create cusparse descriptors
			 */
			cusparseSpMatDescr_t matB;
			CUSPARSE_CHECK(cusparsecreatecsr(&matB, n_idx_rows, n_idx_cols, idxNNZ,
					idxIndptr, idxIndices, idxData));

			// cusparseSpGEMM_workEstimation

			cusparseSpGEMMDescr_t spgemmDesc;
			CUSPARSE_CHECK(cusparseSpGemm_createDescr(&spgemmDesc));

			CUSPARSE_CHECK(cusparsespgemm_workestimation(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					CUSPARSE_OPERATION_TRANSPOSE, &alpha, matA, matB, &beta, matC,
					CUSPARSE_SPGEMM_DEFAULT, &workspace_size1, NULL));

			// cusparseSpGEMM_compute
			device_buffer<char> workspace1(allocator, stream, workspace_size1);

		    // ask bufferSize2 bytes for external memory
		    CUSPARSE_CHECK(cusparsespgemm_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		    		CUSPARSE_OPERATION_TRANSPOSE, &alpha, matA, matB, &beta, matC,
		    		CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &workspace_size2, NULL));

			device_buffer<char> workspace2(allocator, stream, workspace_size2);

			// cusparseSpGEMM_compute
		    // compute the intermediate product of A * B
		    CUSPARSE_CHECK(cusparsespgemm_compute(handle, opA, opB,
		                           &alpha, matA, matB, &beta, matC,
		                           CUSPARSE_SPGEMM_DEFAULT,
		                           spgemmDesc, &workspace_size2, workspace2));

		    // get matrix C non-zero entries C_num_nnz1
		    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
		    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1));
		    // allocate matrix C

		    // update matC with the new pointers
		    CUSPARSE_CHECK(cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values));

		    // copy the final products to the matrix C
		    CUSPARSE_CHECK(cusparsespgemm_copy(handle, opA, opB,
		                        &alpha, matA, matB, &beta, matC,
		                        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

		    workspace1.release(stream);
		    workspace2.release(stream);

		    idx_batch_indptr.release(stream);
		    idx_batch_indices.release(stream);
		    idx_batch_data.release(stream);

		    device_buffer<value_t> C_dense(allocator, stream, C_num_rows1*C_num_cols1);

			// cusparseScsr2dense
		    CUSPARSE_CHECK(cusparsecsr2dense(handle, C_num_rows_1, C_num_cols_1, matC,
		    		dC_csrOffsets, dC_columns, dC_values, C_dense.data(), C_num_cols_1));

		    device_buffer<value_idx> batch_indices(allocator, stream, C_num_rows1*C_num_cols1);
		    device_buffer<value_t> batch_dists(allocator, stream, C_num_rows1*C_num_cols1);

		    // even numbers take bottom, odd numbers take top, merging until end of loop,
		    // where output matrix is populated.
		    size_t merge_buffer_offset = j % 2 == 0 ? 0 : n_query_rows * k;
		    T dists_merge_buffer_ptr = merge_buffer_dists.data()+merge_buffer_offset;
		    int64_t indices_merge_buffer_ptr = merge_buffer_indices.data()+merge_buffer_offset;

		    size_t merge_buffer_tmp_out = n_query_rows * k * 2;
		    T *dists_merge_buffer_tmp_ptr = merge_buffer_dists.data() + merge_buffer_tmp_out;
		    int64_t *indices_merge_buffer_tmp_ptr = merge_buffer_indices.data() + merge_buffer_tmp_out;

		    // build translation buffer to shift resulting indices by the batch

		    std::vector<int64_t> id_ranges(2);
		    id_ranges[0] = idx_batch_start > batch_size ? idx_batch_start - batch_size : 0;
		    id_ranges[1] = idx_batch_start;

		    // kernel to slice first (min) k cols and copy into batched merge buffer
		    select_k(batch_dists.data(),
		    		batch_indices.data(),
		    		C_num_rows1, C_num_cols1,
		    		dists_merge_buffer_ptr,
		    		indices_merge_buffer_ptr,
		    		ascending,
		    		k,
		            stream,
		            /*translation for current batch*/
		            id_ranges[1]);

		    device_buffer<value_idx> batch_indices(allocator, stream, C_num_rows1*C_num_cols1);
		    device_buffer<value_t> batch_dists(allocator, stream, C_num_rows1*C_num_cols1);

		    // combine merge buffers only if there's more than 1 partition to combine
		    Selection::knn_merge_parts(dists_merge_buffer_ptr, indices_merge_buffer_ptr, dists_merge_buffer_tmp_ptr,
		                                indices_merge_buffer_tmp_ptr, C_num_rows1, 2, k,
		                                stream, id_ranges.data());

		    // copy merged output back into merge buffer partition for next iteration
		    copyAsync(indices_merge_buffer_ptr, indices_merge_buffer_tmp_ptr, C_num_rows1, k);
		    copyAsync(dists_merge_buffer_ptr, dists_merge_buffer_tmp_ptr, C_num_rows1, k);
		}

		// Copy final merged batch to output array
		copyAsync(output_indices, indices_merge_buffer_ptr, query_batch_start * k);
		copyAsync(output_dists, dists_merge_buffer_ptr, query_batch_start * k);
	}
}
}; // END namespace Selection
}; // END namespace Sparse
}; // END namespace MLCommon
