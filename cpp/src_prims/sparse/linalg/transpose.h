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

#include <cusparse_v2.h>

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "../utils.h"

namespace raft {
namespace sparse {
namespace linalg {

/**
 * Transpose a set of CSR arrays into a set of CSC arrays.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR data array
 * @param[in] handle : used for invoking cusparse
 * @param[in] csr_indptr : CSR row index array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[out] csc_indptr : CSC row index array
 * @param[out] csc_indices : CSC column indices array
 * @param[out] csc_data : CSC data array
 * @param[in] csr_nrows : Number of rows in CSR
 * @param[in] csr_ncols : Number of columns in CSR
 * @param[in] nnz : Number of nonzeros of CSR
 * @param[in] allocator : Allocator for intermediate memory
 * @param[in] stream : Cuda stream for ordering events
 */
template <typename value_idx, typename value_t>
void csr_transpose(cusparseHandle_t handle, const value_idx *csr_indptr,
                   const value_idx *csr_indices, const value_t *csr_data,
                   value_idx *csc_indptr, value_idx *csc_indices,
                   value_t *csc_data, value_idx csr_nrows, value_idx csr_ncols,
                   value_idx nnz,
                   std::shared_ptr<raft::mr::device::allocator> allocator,
                   cudaStream_t stream) {
  size_t convert_csc_workspace_size = 0;

  CUSPARSE_CHECK(raft::sparse::cusparsecsr2csc_bufferSize(
    handle, csr_nrows, csr_ncols, nnz, csr_data, csr_indptr, csr_indices,
    csc_data, csc_indptr, csc_indices, CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
    &convert_csc_workspace_size, stream));

  raft::mr::device::buffer<char> convert_csc_workspace(
    allocator, stream, convert_csc_workspace_size);

  CUSPARSE_CHECK(raft::sparse::cusparsecsr2csc(
    handle, csr_nrows, csr_ncols, nnz, csr_data, csr_indptr, csr_indices,
    csc_data, csc_indptr, csc_indices, CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
    convert_csc_workspace.data(), stream));
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft