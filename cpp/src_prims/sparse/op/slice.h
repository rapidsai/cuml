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
#include <raft/linalg/unary_op.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "../utils.h"

namespace raft {
namespace sparse {
namespace op {

/**
 * Slice consecutive rows from a CSR array and populate newly sliced indptr array
 * @tparam value_idx
 * @param[in] start_row : beginning row to slice
 * @param[in] stop_row : ending row to slice
 * @param[in] indptr : indptr of input CSR to slice
 * @param[out] indptr_out : output sliced indptr to populate
 * @param[in] start_offset : beginning column offset of input indptr
 * @param[in] stop_offset : ending column offset of input indptr
 * @param[in] stream : cuda stream for ordering events
 */
template <typename value_idx>
void csr_row_slice_indptr(value_idx start_row, value_idx stop_row,
                          const value_idx *indptr, value_idx *indptr_out,
                          value_idx *start_offset, value_idx *stop_offset,
                          cudaStream_t stream) {
  raft::update_host(start_offset, indptr + start_row, 1, stream);
  raft::update_host(stop_offset, indptr + stop_row + 1, 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx s_offset = *start_offset;

  // 0-based indexing so we need to add 1 to stop row. Because we want n_rows+1,
  // we add another 1 to stop row.
  raft::copy_async(indptr_out, indptr + start_row, (stop_row + 2) - start_row,
                   stream);

  raft::linalg::unaryOp<value_idx>(
    indptr_out, indptr_out, (stop_row + 2) - start_row,
    [s_offset] __device__(value_idx input) { return input - s_offset; },
    stream);
}

/**
 * Slice rows from a CSR, populate column and data arrays
 * @tparam value_idx : data type of CSR index arrays
 * @tparam value_t : data type of CSR data array
 * @param[in] start_offset : beginning column offset to slice
 * @param[in] stop_offset : ending column offset to slice
 * @param[in] indices : column indices array from input CSR
 * @param[in] data : data array from input CSR
 * @param[out] indices_out : output column indices array
 * @param[out] data_out : output data array
 * @param[in] stream : cuda stream for ordering events
 */
template <typename value_idx, typename value_t>
void csr_row_slice_populate(value_idx start_offset, value_idx stop_offset,
                            const value_idx *indices, const value_t *data,
                            value_idx *indices_out, value_t *data_out,
                            cudaStream_t stream) {
  raft::copy(indices_out, indices + start_offset, stop_offset - start_offset,
             stream);
  raft::copy(data_out, data + start_offset, stop_offset - start_offset, stream);
}

};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft