/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>

#include <cusparse_v2.h>
#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <label/classlabels.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>
#include <sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace op {

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row.
 *
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_sort(int m, int n, int nnz, int *rows, int *cols, T *vals,
              std::shared_ptr<MLCommon::deviceAllocator> d_alloc,
              cudaStream_t stream) {
  cusparseHandle_t handle = NULL;

  size_t pBufferSizeInBytes = 0;

  CUSPARSE_CHECK(cusparseCreate(&handle));
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, rows, cols,
                                                &pBufferSizeInBytes));

  MLCommon::device_buffer<int> d_P(d_alloc, stream, nnz);
  MLCommon::device_buffer<char> pBuffer(d_alloc, stream, pBufferSizeInBytes);

  CUSPARSE_CHECK(cusparseCreateIdentityPermutation(handle, nnz, d_P.data()));

  CUSPARSE_CHECK(cusparseXcoosortByRow(handle, m, n, nnz, rows, cols,
                                       d_P.data(), pBuffer.data()));

  MLCommon::device_buffer<T> vals_sorted(d_alloc, stream, nnz);

  CUSPARSE_CHECK(raft::sparse::cusparsegthr<T>(
    handle, nnz, vals, vals_sorted.data(), d_P.data(), stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  raft::copy(vals, vals_sorted.data(), nnz, stream);

  CUSPARSE_CHECK(cusparseDestroy(handle));
}

/**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param d_alloc device allocator for temporary buffers
 * @param stream: the cuda stream to use
 */
template <typename T>
void coo_sort(COO<T> *const in,
              std::shared_ptr<MLCommon::deviceAllocator> d_alloc,
              cudaStream_t stream) {
  coo_sort<T>(in->n_rows, in->n_cols, in->nnz, in->rows(), in->cols(),
              in->vals(), d_alloc, stream);
}
};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft