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

#include <matrix/reverse.cuh>
#include <matrix/matrix.cuh>

#include <selection/columnWiseSort.cuh>
#include <selection/knn.cuh>

#include "cusparse_wrappers.h"
#include <common/device_buffer.hpp>
#include <common/cudart_tools.h>
#include <cuda_utils.cuh>



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

	copy(indptr_out, indptr+start_row, indptr+stop_row+1, stream);
}

template<typename value_idx, typename value_t>
void csr_row_slice_populate(value_idx start_offset, value_idx stop_offset,
		const value_idx *indices, const value_t *data,
		value_idx *indices_out, value_t *data_out, cudaStream_t stream) {

	copy(indices_out, indices+start_offset, indices+stop_offset, stream);
	copy(data_out, data+start_offset, data+stop_offset, stream);
}

template<typename T>
__global__ void select_k(T *out, const T *in, size_t n_rows, size_t n_cols, int k) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(i < k * n_rows) {
		size_t row = i / k;
		size_t col = i % k;

		out[row * k + col]  = in[row * n_cols + col];
	}
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
template <typename value_idx = int, typename value_t, int TPB_X=32>
void brute_force_knn(const value_idx *idxIndptr,
					 const value_idx *idxIndices,
					 const value_t *idxData,
					 value_idx idxNNZ,
					 value_idx n_idx_rows, value_idx n_idx_cols,
					 const value_idx *queryIndptr,
					 const value_idx *queryIndices,
					 const value_t *queryData,
					 value_idx queryNNZ,
					 value_idx n_query_rows, value_idx n_query_cols,
					 value_idx *output_indices,
					 value_t *output_dists,
					 int k,
					 int batch_size = 2<<20, // approx 1M
					 cusparseHandle_t cusparseHandle,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t stream,
                     std::vector<int64_t> *translations = nullptr,
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false) {

	int n_batches = ceilDiv(n_idx_rows/batch_size);
	bool ascending = true;
	if(metric == ML::MetricType::METRIC_INNER_PRODUCT)
		ascending = false;

	value_t alpha = 1.0, beta = 0.0;

	// TODO: Should first batch over query, then over index

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

		device_buffer<int64_t> merge_buffer_indices(allocator, stream, k * n_query_rows * 2);
		device_buffer<T> merge_buffer_dists(allocator, stream, k * n_query_rows * 2);

		for(int j = 0; j < n_batches_idx; j++) {

			/**
			 * Compute query batch info
			 */
			value_idx idx_batch_start = i * batch_size;
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

		    device_buffer<value_t> batch_indices(allocator, stream, C_num_rows1*C_num_cols1);
		    device_buffer<value_t> batch_dists(allocator, stream, C_num_rows1*C_num_cols1);

			// sortColumnsPerRow
		    size_t sortCols_workspacesize;
		    Selection::sortColumnsPerRow(C_dense.data(), batch_dists.data(), C_num_rows1,
		                      C_num_cols1, true, nullptr, &sortCols_workspacesize, stream,
		                      batch_indices.data(), ascending);

		    device_buffer<char> sortCols_workspace(allocator, stream, sortCols_workspacesize);

		    Selection::sortColumnsPerRow(C_dense.data(), batch_dists.data(), C_num_rows1,
		                      C_num_cols1, true, sortCols_workspace, &sortCols_workspacesize, stream,
		                      batch_indices.data(), ascending);

		    // kernel to slice first (min) k cols and copy into batched merge buffer

		    size_t merge_buffer_offset = j % 2 == 0 ? 0 : n_query_rows * k;
		    T *dists_merge_buffer_ptr = merge_buffer_dists.data()+merge_buffer_offset;
		    int64_t *indices_merge_buffer_ptr = merge_buffer_indices.data()+merge_buffer_offset;

		    select_k<<<ceilDiv(k*C_num_rows1, TPB_X), TPB_X, 0, stream>>>(
		    		dists_merge_buffer_ptr, batch_dists.data(), C_num_rows1, C_num_cols1, k);
		    select_k<<<ceilDiv(k*C_num_rows1, TPB_X), TPB_X, 0, stream>>>(
		    		indices_merge_buffer_ptr, batch_indices.data(), C_num_rows1, C_num_cols1, k);

			// knn_merge_parts
		    // Merge parts only executed for batch > 1
		    // even numbers take bottom, odd numbers take top, merging until end of loop, where output matrix is populated.
		    Selection::knn_merge_parts(dists_merge_buffer_ptr, indices_merge_buffer_ptr, float *outK,
		                                int64_t *outV, size_t n_samples, int n_parts, int k,
		                                cudaStream_t stream, int64_t *translations)

		}

		// Copy final merged batch to output array
	}
}
}
} // END namespace MLCommon
