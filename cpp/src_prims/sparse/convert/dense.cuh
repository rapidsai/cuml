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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "../utils.h"

namespace raft {
namespace sparse {
namespace convert {

template <typename value_t>
__global__ void csr_to_dense_warp_per_row_kernel(int n_cols,
                                                 const value_t *csrVal,
                                                 const int *csrRowPtr,
                                                 const int *csrColInd,
                                                 value_t *a) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  int colStart = csrRowPtr[row];
  int colEnd = csrRowPtr[row + 1];
  int rowNnz = colEnd - colStart;

  for (int i = tid; i < rowNnz; i += blockDim.x) {
    int colIdx = colStart + i;
    if (colIdx < colEnd) {
      int col = csrColInd[colIdx];
      a[row * n_cols + col] = csrVal[colIdx];
    }
  }
}

/**
 * Convert CSR arrays to a dense matrix in either row-
 * or column-major format. A custom kernel is used when
 * row-major output is desired since cusparse does not
 * output row-major.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR value array
 * @param[in] handle : cusparse handle for conversion
 * @param[in] nrows : number of rows in CSR
 * @param[in] ncols : number of columns in CSR
 * @param[in] csr_indptr : CSR row index pointer array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[in] lda : Leading dimension (used for col-major only)
 * @param[out] out : Dense output array of size nrows * ncols
 * @param[in] stream : Cuda stream for ordering events
 * @param[in] row_major : Is row-major output desired?
 */
template <typename value_idx, typename value_t>
void csr_to_dense(cusparseHandle_t handle, value_idx nrows, value_idx ncols,
                  const value_idx *csr_indptr, const value_idx *csr_indices,
                  const value_t *csr_data, value_idx lda, value_t *out,
                  cudaStream_t stream, bool row_major = true) {
  if (!row_major) {
    /**
     * If we need col-major, use cusparse.
     */
    cusparseMatDescr_t out_mat;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&out_mat));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(out_mat, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(out_mat, CUSPARSE_MATRIX_TYPE_GENERAL));

    CUSPARSE_CHECK(raft::sparse::cusparsecsr2dense(
      handle, nrows, ncols, out_mat, csr_data, csr_indptr, csr_indices, out,
      lda, stream));

    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(out_mat));

  } else {
    int blockdim = block_dim(ncols);
    CUDA_CHECK(
      cudaMemsetAsync(out, 0, nrows * ncols * sizeof(value_t), stream));
    csr_to_dense_warp_per_row_kernel<<<nrows, blockdim, 0, stream>>>(
      ncols, csr_data, csr_indptr, csr_indices, out);
  }
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft