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
#include <raft/distance/detail/matrix/matrix.hpp>
#include <raft/util/cuda_utils.cuh>

#include <thrust/transform_scan.h>

namespace ML {
namespace SVM {

using namespace raft::distance::matrix::detail;

/*
 * Extension to raft matrix wrapper that owns the backing memory
 * and allows dynamic resizing. This simplifies CSR row extraction
 * where the target nnz is not known in advance
 */
template <typename math_t>
class ResizableCsrMatrix : public CsrMatrix<math_t> {
 public:
  ResizableCsrMatrix(int rows, int cols, int nnz, cudaStream_t stream)
    : CsrMatrix<math_t>(nullptr, nullptr, nullptr, nnz, rows, cols),
      d_indptr(rows + 1, stream),
      d_indices(nnz, stream),
      d_data(nnz, stream)
  {
    CsrMatrix<math_t>::indptr  = d_indptr.data();
    CsrMatrix<math_t>::indices = d_indices.data();
    CsrMatrix<math_t>::data    = d_data.data();
  }

  void reserveRows(int rows, cudaStream_t stream)
  {
    d_indptr.reserve(rows + 1, stream);
    CsrMatrix<math_t>::indptr = d_indptr.data();
  }

  void reserveNnz(int nnz, cudaStream_t stream)
  {
    d_indices.reserve(nnz, stream);
    d_data.reserve(nnz, stream);
    CsrMatrix<math_t>::indices = d_indices.data();
    CsrMatrix<math_t>::data    = d_data.data();
  }

  rmm::device_uvector<int> d_indptr, d_indices;
  rmm::device_uvector<math_t> d_data;
};

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
void matrixRowNorm(const Matrix<math_t>& matrix,
                   math_t* target,
                   raft::linalg::NormType norm,
                   cudaStream_t stream)
{
  if (matrix.isDense()) {
    int minor = matrix.asDense()->is_row_major ? matrix.n_cols : matrix.n_rows;
    ASSERT(matrix.asDense()->ld == minor, "Dense matrix rowNorm only support contiguous data");
    raft::linalg::rowNorm(target,
                          matrix.asDense()->data,
                          matrix.n_cols,
                          matrix.n_rows,
                          norm,
                          matrix.asDense()->is_row_major,
                          stream);
  } else {
    auto csr_matrix = matrix.asCsr();
    raft::sparse::linalg::rowNormCsr(
      target, csr_matrix->indptr, csr_matrix->data, csr_matrix->nnz, matrix.n_rows, norm, stream);
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
static void extractRows(const Matrix<math_t>& matrix_in,
                        Matrix<math_t>& matrix_out,
                        const int* row_indices,
                        int num_indices,
                        cudaStream_t stream)
{
  matrix_out.n_rows = num_indices;
  matrix_out.n_cols = matrix_in.n_cols;

  // initialize dense target
  if (!matrix_in.isDense() && matrix_out.isDense()) {
    thrust::device_ptr<math_t> output_ptr(matrix_out.asDense()->data);
    thrust::fill(thrust::cuda::par.on(stream),
                 output_ptr,
                 output_ptr + num_indices * matrix_in.n_cols,
                 (math_t)0);
  }

  if (matrix_in.isDense()) {
    ASSERT(matrix_out.isDense(), "Cannot extract sparse rows from dense matrix");
    auto dense_matrix_in  = matrix_in.asDense();
    auto dense_matrix_out = matrix_out.asDense();
    int minor             = dense_matrix_in->is_row_major ? matrix_in.n_cols : matrix_in.n_rows;
    ASSERT(minor == dense_matrix_in->ld, "No padding supported");
    dense_matrix_out->is_row_major = dense_matrix_in->is_row_major;
    dense_matrix_out->ld = dense_matrix_out->is_row_major ? matrix_out.n_cols : matrix_out.n_rows;

    raft::matrix::copyRows<math_t, int, size_t>(matrix_in.asDense()->data,
                                                matrix_in.n_rows,
                                                matrix_in.n_cols,
                                                matrix_out.asDense()->data,
                                                row_indices,
                                                num_indices,
                                                stream,
                                                false);
  } else {
    auto csr_matrix_in = matrix_in.asCsr();
    int* indptr        = csr_matrix_in->indptr;
    int* indices       = csr_matrix_in->indices;
    math_t* data       = csr_matrix_in->data;

    // copy with 1 warp per row for now, blocksize 256
    const dim3 bs(32, 8, 1);
    const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
    if (matrix_out.isDense()) {
      auto dense_matrix_out          = matrix_out.asDense();
      dense_matrix_out->is_row_major = false;
      dense_matrix_out->ld           = num_indices;
      extractDenseRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(
        dense_matrix_out->data, indptr, indices, data, row_indices, num_indices);
    } else {
      ResizableCsrMatrix<math_t>* csr_matrix_out =
        dynamic_cast<ResizableCsrMatrix<math_t>*>(&matrix_out);
      ASSERT(csr_matrix_out != nullptr, "Matrix csr target should be resizable.");

      csr_matrix_out->reserveRows(num_indices, stream);

      // row sizes + inclusive_scan -> row end positions
      thrust::device_ptr<int> row_sizes_ptr(
        csr_matrix_out->indptr);  // store size in target for now
      thrust::device_ptr<const int> row_new_indices_ptr(row_indices);
      // thrust::fill(thrust::cuda::par.on(stream), row_sizes_ptr, row_sizes_ptr + 1, 0); // will be
      // done inside kernel
      thrust::transform_inclusive_scan(thrust::cuda::par.on(stream),
                                       row_new_indices_ptr,
                                       row_new_indices_ptr + num_indices,
                                       row_sizes_ptr + 1,
                                       rowsize(indptr),
                                       thrust::plus<int>());

      cudaDeviceSynchronize();

      // retrieve nnz from indptr[num_indices]
      raft::update_host(&(csr_matrix_out->nnz), csr_matrix_out->indptr + num_indices, 1, stream);

      csr_matrix_out->reserveNnz(csr_matrix_out->nnz, stream);

      extractCSRRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(csr_matrix_out->indptr,
                                                           csr_matrix_out->indices,
                                                           csr_matrix_out->data,
                                                           indptr,
                                                           indices,
                                                           data,
                                                           row_indices,
                                                           num_indices);
    }
  }
  cudaDeviceSynchronize();
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace SVM
}  // namespace ML
