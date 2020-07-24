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
template <typename value_idx = int, typename value_t>
void brute_force_knn(const value_idx *idxIndptr, const value_idx *idxIndices, const value_t *idxData,
		value_idx idxNNZ, value_idx n_idx_rows,
					 const value_idx *queryIndptr, const value_idx *queryIndices, const value_t *queryData,
					 value_idx queryNNZ, value_idx n_query_rows,
					 value_idx *output_indices, value_t *output_dists, int k,
					 int batch_size = 2<<20, // approx 1M
					 cusparseHandle_t cusparseHandle,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t stream,
                     std::vector<int64_t> *translations = nullptr,
                     ML::MetricType metric = ML::MetricType::METRIC_L2,
                     float metricArg = 0, bool expanded_form = false) {

	int n_batches = ceilDiv(n_idx_rows/batch_size);

	for(int i = 0; i < n_batches_idx; i++) {

		/**
		 * Compute index batch info
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

		for(int j = 0; j < n_batches_query; j++) {

			// Perform row-slice of query

			// cusparseSpGEMM_workEstimation
			// cusparseSpGEMM_compute
			// cusparseSpGEMM_compute
			// cusparseSpGEMM_copy
			// cusparseScsr2dense

			// sortColumnsPerRow

			// knn_merge_parts
		}

		// Copy final merged batch to output array
	}
}
}
} // END namespace MLCommon
