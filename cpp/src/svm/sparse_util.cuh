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
#include <cuml/matrix/cumlmatrix.hpp>
#include <raft/util/cuda_utils.cuh>
#include <thrust/transform_scan.h>

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
static __global__ void extractCSRRowsFromCSR(int* indptr_out,  // already holds start positions
                                             int* indices_out,
                                             math_t* data_out,
                                             const int* indptr_in,
                                             const int* indices_in,
                                             const math_t* data_in,
                                             const int* row_indices,
                                             const int num_indices)
{
  assert(gridDim.y == 1 && gridDim.z == 1);
  // all threads in x-direction are responsible for one line of csr
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx >= num_indices) return;

  int row_idx = row_indices[idx];

  int in_offset  = indptr_in[row_idx];
  int row_length = indptr_in[row_idx + 1] - in_offset;
  int out_offset = indptr_out[idx];
  for (int pos = threadIdx.x; pos < row_length; pos += blockDim.x) {
    indices_out[out_offset + pos] = indices_in[in_offset + pos];
    data_out[out_offset + pos]    = data_in[in_offset + pos];
  }

  // the last row has to set the indptr_out[num_indices] value
  // as it was not a start position
  if (threadIdx.x == 0 && idx == num_indices - 1) {
    indptr_out[num_indices] = out_offset + row_length;
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

struct rowsize : public thrust::unary_function<int, int> {
  const int* indptr_;
  rowsize(const int* indptr) : indptr_(indptr) {}

  __device__ int64_t operator()(const int& x) const { return indptr_[x + 1] - indptr_[x]; }
};

template <typename math_t>
static void copySparseRowsToMatrix(const int* indptr,
                                   const int* indices,
                                   const math_t* data,
                                   int n_rows,
                                   int n_cols,
                                   MLCommon::Matrix::Matrix<math_t>& matrix_out,
                                   const int* row_indices,
                                   int num_indices,
                                   cudaStream_t stream)
{
  // TODO
  // matrix_out should to be allocated in advance to prevent costly allocs
  // however, nnz size cannot be acurately guessed in advance
  // we should introduce a fallback in case data is too large to fit
  // i.e. return error to allow caller to increase buffer

  // initialize dense target
  if (matrix_out.getType() == MLCommon::Matrix::MatrixType::DENSE) {
    thrust::device_ptr<math_t> output_ptr(matrix_out.asDense()->data);
    thrust::fill(
      thrust::cuda::par.on(stream), output_ptr, output_ptr + num_indices * n_cols, (math_t)0);
  }

  matrix_out.n_rows = num_indices;
  matrix_out.n_cols = n_cols;

  // copy with 1 warp per row for now, blocksize 256
  const dim3 bs(32, 8, 1);
  const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
  switch (matrix_out.getType()) {
    case MLCommon::Matrix::MatrixType::DENSE: {
      extractDenseRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(
        matrix_out.asDense()->data, indptr, indices, data, row_indices, num_indices);
      break;
    }
    case MLCommon::Matrix::MatrixType::CSR: {
      MLCommon::Matrix::CsrMatrix<math_t>* csr_out = matrix_out.asCsr();

      // row sizes + exclusive_scan -> row start positions
      thrust::device_ptr<int> row_sizes_ptr(csr_out->indptr);  // store size in target for now
      thrust::device_ptr<const int> row_new_indices_ptr(row_indices);
      thrust::transform_exclusive_scan(row_new_indices_ptr,
                                       row_new_indices_ptr + num_indices,
                                       row_sizes_ptr,
                                       rowsize(indptr),
                                       0,
                                       thrust::plus<int>());

      extractCSRRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(csr_out->indptr,
                                                           csr_out->indices,
                                                           csr_out->data,
                                                           indptr,
                                                           indices,
                                                           data,
                                                           row_indices,
                                                           num_indices);

      // retrieve nnz from indptr[num_indices]
      raft::update_host(&(csr_out->nnz), csr_out->indptr + num_indices, 1, stream);
      break;
    }
    default: THROW("Solve not implemented for matrix type %d", matrix_out.getType());
  }
  cudaDeviceSynchronize();
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace SVM
}  // namespace ML
