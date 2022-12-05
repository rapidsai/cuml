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
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace SVM {

template <typename math_t>
static __global__ void extractDenseRowsFromCSR(math_t* out,
                                               const int* indptr,
                                               const int* indices,
                                               const math_t* data,
                                               const int* row_indices,
                                               const int num_indices)
{
  assert(gridDim.y == 1 && gridDim.z == 1);
  // all threads in x-direction are responsible for one line of csr
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx >= num_indices) return;

  int row_idx = row_indices[idx];

  int rowStartIdx = indptr[row_idx];
  int rowEndIdx   = indptr[row_idx + 1];
  for (int pos = rowStartIdx + threadIdx.x; pos < rowEndIdx; pos += blockDim.x) {
    int col_idx                      = indices[pos];
    out[idx + col_idx * num_indices] = data[pos];
  }
}

template <typename math_t>
static void copySparseRowsToDense(const int* indptr,
                                  const int* indices,
                                  const math_t* data,
                                  int n_rows,
                                  int n_cols,
                                  math_t* output,
                                  const int* row_indices,
                                  int num_indices,
                                  cudaStream_t stream)
{
  thrust::device_ptr<math_t> output_ptr(output);
  thrust::fill(
    thrust::cuda::par.on(stream), output_ptr, output_ptr + num_indices * n_cols, (math_t)0);

  // copy with 1 warp per row for now, blocksize 256
  const dim3 bs(32, 8, 1);
  const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
  extractDenseRowsFromCSR<math_t>
    <<<gs, bs, 0, stream>>>(output, indptr, indices, data, row_indices, num_indices);
  cudaDeviceSynchronize();
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace SVM
}  // namespace ML
