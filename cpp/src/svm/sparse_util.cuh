/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuml/matrix/matrix.h>
#include <raft/distance/kernels.cuh>
#include <raft/util/cuda_utils.cuh>
#include <thrust/transform_scan.h>

namespace ML {
namespace SVM {

// kernel call helper
template <typename math_t>
void KernelOp(raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              const MLCommon::Matrix::Matrix<math_t>& input1,
              const MLCommon::Matrix::Matrix<math_t>& input2,
              MLCommon::Matrix::DenseMatrix<math_t>& result,
              const raft::handle_t& handle,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto gram_result = result.get_dense_view();
  if (input1.is_dense()) {
    assert(input2.is_dense());
    // Dense x Dense
    auto dense1 = input1.as_dense()->get_const_dense_view();
    auto dense2 = input2.as_dense()->get_const_dense_view();
    (*kernel)(handle, dense1, dense2, gram_result, norm1, norm2);
  } else {
    auto csr1 = input1.as_csr()->get_const_csr_view();
    if (input2.is_dense()) {
      auto dense2 = input2.as_dense()->get_const_dense_view();
      (*kernel)(handle, csr1, dense2, gram_result, norm1, norm2);
    } else {
      auto csr2 = input2.as_csr()->get_const_csr_view();
      (*kernel)(handle, csr1, csr2, gram_result, norm1, norm2);
    }
  }
}

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
static __global__ void extractCSRRowsFromCSR(int* indptr_out,  // already holds end positions
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

  // the first row has to set the indptr_out[0] value to 0
  if (threadIdx.x == 0 && idx == 0) { indptr_out[0] = 0; }

  int row_idx = row_indices[idx];

  int in_offset  = indptr_in[row_idx];
  int row_length = indptr_in[row_idx + 1] - in_offset;
  int out_offset = indptr_out[idx];
  for (int pos = threadIdx.x; pos < row_length; pos += blockDim.x) {
    indices_out[out_offset + pos] = indices_in[in_offset + pos];
    data_out[out_offset + pos]    = data_in[in_offset + pos];
  }
}

template <typename math_t>
void matrixRowNorm(const raft::handle_t& handle,
                   const MLCommon::Matrix::Matrix<math_t>& matrix,
                   math_t* target,
                   raft::linalg::NormType norm)
{
  if (matrix.is_dense()) {
    int minor = matrix.as_dense()->is_row_major() ? matrix.get_n_cols() : matrix.get_n_rows();
    ASSERT(matrix.as_dense()->get_ld() == minor,
           "Dense matrix rowNorm only support contiguous data");
    raft::linalg::rowNorm(target,
                          matrix.as_dense()->get_data(),
                          matrix.get_n_cols(),
                          matrix.get_n_rows(),
                          norm,
                          matrix.as_dense()->is_row_major(),
                          handle.get_stream());
  } else {
    auto csr_matrix = matrix.as_csr();
    raft::sparse::linalg::rowNormCsr(handle,
                                     csr_matrix->get_indptr(),
                                     csr_matrix->get_data(),
                                     csr_matrix->get_nnz(),
                                     matrix.get_n_rows(),
                                     target,
                                     norm);
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
static void extractRows(const MLCommon::Matrix::Matrix<math_t>& matrix_in,
                        MLCommon::Matrix::Matrix<math_t>& matrix_out,
                        const int* row_indices,
                        int num_indices,
                        const raft::handle_t& handle)
{
  auto stream = handle.get_stream();
  matrix_out.initialize_dimensions(handle, num_indices, matrix_in.get_n_cols());
  // initialize dense target
  if (!matrix_in.is_dense() && matrix_out.is_dense()) {
    thrust::device_ptr<math_t> output_ptr(matrix_out.as_dense()->get_data());
    thrust::fill(thrust::cuda::par.on(stream),
                 output_ptr,
                 output_ptr + matrix_out.get_n_rows() * matrix_out.get_n_cols(),
                 (math_t)0);
  }

  if (matrix_in.is_dense()) {
    ASSERT(matrix_out.is_dense(), "Cannot extract sparse rows from dense matrix");
    auto dense_matrix_in  = matrix_in.as_dense();
    auto dense_matrix_out = matrix_out.as_dense();
    int minor = dense_matrix_in->is_row_major() ? matrix_in.get_n_cols() : matrix_in.get_n_rows();
    ASSERT(minor == dense_matrix_in->get_ld(), "No padding supported");
    ASSERT(dense_matrix_out->is_row_major() == dense_matrix_in->is_row_major(),
           "Layout does not match");

    raft::matrix::copyRows<math_t, int, size_t>(matrix_in.as_dense()->get_data(),
                                                matrix_in.get_n_rows(),
                                                matrix_in.get_n_cols(),
                                                matrix_out.as_dense()->get_data(),
                                                row_indices,
                                                num_indices,
                                                stream,
                                                dense_matrix_in->is_row_major());
  } else {
    auto csr_matrix_in = matrix_in.as_csr();
    int* indptr        = csr_matrix_in->get_indptr();
    int* indices       = csr_matrix_in->get_indices();
    math_t* data       = csr_matrix_in->get_data();

    // copy with 1 warp per row for now, blocksize 256
    const dim3 bs(32, 8, 1);
    const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
    if (matrix_out.is_dense()) {
      auto dense_matrix_out = matrix_out.as_dense();
      ASSERT(!dense_matrix_out->is_row_major() && dense_matrix_out->get_ld() == num_indices,
             "Invalid Layout");
      extractDenseRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(
        dense_matrix_out->get_data(), indptr, indices, data, row_indices, num_indices);
    } else {
      auto csr_matrix_out = matrix_out.as_csr();

      // row sizes + inclusive_scan -> row end positions
      thrust::device_ptr<int> row_sizes_ptr(csr_matrix_out->get_indptr());
      thrust::device_ptr<const int> row_new_indices_ptr(row_indices);
      thrust::transform_inclusive_scan(thrust::cuda::par.on(stream),
                                       row_new_indices_ptr,
                                       row_new_indices_ptr + num_indices,
                                       row_sizes_ptr + 1,
                                       rowsize(indptr),
                                       thrust::plus<int>());

      cudaStreamSynchronize(stream);

      // retrieve nnz from indptr[num_indices]
      int nnz;
      raft::update_host(&nnz, csr_matrix_out->get_indptr() + num_indices, 1, stream);

      csr_matrix_out->initialize_sparsity(handle, nnz);

      extractCSRRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(csr_matrix_out->get_indptr(),
                                                           csr_matrix_out->get_indices(),
                                                           csr_matrix_out->get_data(),
                                                           indptr,
                                                           indices,
                                                           data,
                                                           row_indices,
                                                           num_indices);
    }
  }
  cudaStreamSynchronize(stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace SVM
}  // namespace ML
