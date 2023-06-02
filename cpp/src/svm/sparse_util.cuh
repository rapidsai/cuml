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
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_scan.h>

namespace ML {
namespace SVM {

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * DENSE(mdspan) x DENSE(mdspan) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::device_matrix_view<math_t, int, raft::layout_stride> input1,
              raft::device_matrix_view<math_t, int, raft::layout_stride> input2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto const_input1 = raft::make_const_mdspan(input1);
  auto const_input2 = raft::make_const_mdspan(input2);
  auto result_view  = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    result, input1.extent(0), input2.extent(0), 0);
  (*kernel)(handle, const_input1, const_input2, result_view, norm1, norm2);
}

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * DENSE(mdspan) x DENSE(raw pointer) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [in] rows2 number of rows for input2
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::device_matrix_view<math_t, int, raft::layout_stride> input1,
              math_t* input2,
              int rows2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto view2 = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input2, rows2, input1.extent(1), 0);
  KernelOp(handle, kernel, input1, view2, result, norm1, norm2);
}

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * DENSE(raw pointer) x DENSE(raw pointer) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] rows1 number of rows for input1
 * @param [in] cols number of cols for input1/input2
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [in] rows2 number of rows for input2
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              math_t* input1,
              int rows1,
              int cols,
              math_t* input2,
              int rows2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto view1 = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input1, rows1, cols, 0);
  auto view2 = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input2, rows2, cols, 0);
  KernelOp(handle, kernel, view1, view2, result, norm1, norm2);
}

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * CSR(matrix_view) x CSR(matrix_view) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::device_csr_matrix_view<math_t, int, int, int> input1,
              raft::device_csr_matrix_view<math_t, int, int, int> input2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto const_input1 = raft::make_device_csr_matrix_view<const math_t, int, int, int>(
    input1.get_elements().data(), input1.structure_view());
  auto const_input2 = raft::make_device_csr_matrix_view<const math_t, int, int, int>(
    input2.get_elements().data(), input2.structure_view());
  auto result_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    result, input1.structure_view().get_n_rows(), input2.structure_view().get_n_rows(), 0);
  (*kernel)(handle, const_input1, const_input2, result_view, norm1, norm2);
}

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * CSR(matrix_view) x DENSE(mdspan) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::device_csr_matrix_view<math_t, int, int, int> input1,
              raft::device_matrix_view<math_t, int, raft::layout_stride> input2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto const_input1 = raft::make_device_csr_matrix_view<const math_t, int, int, int>(
    input1.get_elements().data(), input1.structure_view());
  auto const_input2 = raft::make_const_mdspan(input2);
  auto result_view  = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    result, input1.structure_view().get_n_rows(), input2.extent(0), 0);
  (*kernel)(handle, const_input1, const_input2, result_view, norm1, norm2);
}

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * DENSE(mdspan) x CSR(matrix_view) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::device_matrix_view<math_t, int, raft::layout_stride> input1,
              raft::device_csr_matrix_view<math_t, int, int, int> input2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  ASSERT(false, "KernelOp not implemented for DENSE x CSR.");
}

/**
 * @brief Kernel call helper
 *
 * Specialization for
 * CSR(matrix_view) x DENSE(raw pointer) -> DENSE(raw pointer)
 *
 * @param [in] handle raft handle
 * @param [in] kernel kernel instance
 * @param [in] input1 matrix input, either dense or csr [i, j]
 * @param [in] input2 matrix input, either dense or csr [k, j]
 * @param [in] rows2 number of rows for input2
 * @param [out] result evaluated kernel matrix [i, k]
 * @param [in] norm_x1 L2-norm of input1's rows (optional, only RBF)
 * @param [in] norm_x2 L2-norm of input2's rows (optional, only RBF)
 */
template <typename math_t>
void KernelOp(const raft::handle_t& handle,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::device_csr_matrix_view<math_t, int, int, int> input1,
              math_t* input2,
              int rows2,
              math_t* result,
              math_t* norm1 = nullptr,
              math_t* norm2 = nullptr)
{
  auto view2 = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input2, rows2, input1.structure_view().get_n_cols(), 0);
  KernelOp(handle, kernel, input1, view2, result, norm1, norm2);
}

/**
 * @brief Create view on matrix batch of contiguous rows
 *
 * This specialization creates a device matrix view
 * representing a batch from a from a given device
 * matrix view.
 *
 * @param [in] handle raft handle
 * @param [in] matrix matrix input, csr [i, j]
 * @param [in] batch_size number of rows within batch
 * @param [in] offset row offset for batch start
 * @param [in] host_indptr unused
 * @param [in] target_indptr unused
 * @param [in] stream stream
 * @return matrix_batch
 */
template <typename math_t>
raft::device_matrix_view<math_t, int, raft::layout_stride> getMatrixBatch(
  raft::device_matrix_view<math_t, int, raft::layout_stride> matrix,
  int batch_size,
  int offset,
  int* host_indptr,
  int* target_indptr,
  cudaStream_t stream)
{
  if (batch_size == matrix.extent(0)) {
    return matrix;
  } else {
    return raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
      matrix.data_handle() + offset, batch_size, matrix.extent(1), matrix.extent(0));
  }
}

/**
 * @brief Create view on matrix batch of contiguous rows
 *
 * This specialization creates a device csr matrix view
 * representing a batch from a from a given device csr
 * matrix view.
 *
 * @param [in] handle raft handle
 * @param [in] matrix matrix input, csr [i, j]
 * @param [in] batch_size number of rows within batch
 * @param [in] offset row offset for batch start
 * @param [in] host_indptr host copy of indptr
 * @param [in] target_indptr target buffer for modified indptr
 * @param [in] stream stream
 * @return matrix_batch
 */
template <typename math_t>
raft::device_csr_matrix_view<math_t, int, int, int> getMatrixBatch(
  raft::device_csr_matrix_view<math_t, int, int, int> matrix,
  int batch_size,
  int offset,
  int* host_indptr,
  int* target_indptr,
  cudaStream_t stream)
{
  auto csr_struct_in = matrix.structure_view();
  if (batch_size == csr_struct_in.get_n_rows()) {
    return matrix;
  } else {
    int* indptr_in  = csr_struct_in.get_indptr().data();
    int* indices_in = csr_struct_in.get_indices().data();
    math_t* data_in = matrix.get_elements().data();

    int nnz_offset = host_indptr[offset];
    int batch_nnz  = host_indptr[offset + batch_size] - nnz_offset;
    {
      thrust::device_ptr<int> inptr_src(indptr_in + offset);
      thrust::device_ptr<int> inptr_tgt(target_indptr);
      thrust::transform(thrust::cuda::par.on(stream),
                        inptr_src,
                        inptr_src + batch_size + 1,
                        thrust::make_constant_iterator(nnz_offset),
                        inptr_tgt,
                        thrust::minus<int>());
    }

    auto csr_struct_out = raft::make_device_compressed_structure_view<int, int, int>(
      target_indptr, indices_in + nnz_offset, batch_size, csr_struct_in.get_n_cols(), batch_nnz);
    return raft::make_device_csr_matrix_view(data_in + nnz_offset, csr_struct_out);
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

/**
 * Returns whether MatrixViewType is a device_matrix_view
 *
 * @return true if MatrixViewType is device dense mdspan
 */
template <typename MatrixViewType>
bool isDenseType()
{
  return (std::is_same<MatrixViewType,
                       raft::device_matrix_view<float, int, raft::layout_stride>>::value ||
          std::is_same<MatrixViewType,
                       raft::device_matrix_view<double, int, raft::layout_stride>>::value);
}

/**
 * @brief Specialization of compute row norm for dense matrix
 *
 * This utility runs the row norm computation for a dense and
 * contiguous device matrix.
 *
 *
 * @param [in] handle raft handle
 * @param [in] matrix matrix input, dense [i, j]
 * @param [out] target row norm, size needs to be at least [i]
 * @param [in] norm norm type to be evaluated
 */
template <typename math_t>
void matrixRowNorm(const raft::handle_t& handle,
                   raft::device_matrix_view<math_t, int, raft::layout_stride> matrix,
                   math_t* target,
                   raft::linalg::NormType norm)
{
  bool is_row_major_contiguous = matrix.stride(1) == 1 && matrix.stride(0) == matrix.extent(1);
  bool is_col_major_contiguous = matrix.stride(0) == 1 && matrix.stride(1) == matrix.extent(0);
  ASSERT(is_row_major_contiguous || is_col_major_contiguous,
         "Dense matrix rowNorm only support contiguous data");
  raft::linalg::rowNorm(target,
                        matrix.data_handle(),
                        matrix.extent(1),  //! cols first arg!
                        matrix.extent(0),
                        norm,
                        is_row_major_contiguous,
                        handle.get_stream());
}

/**
 * @brief Specialization of compute row norm for csr matrix
 *
 * This utility runs the row norm computation for a csr matrix.
 *
 * @param [in] handle raft handle
 * @param [in] matrix matrix input, csr [i, j]
 * @param [out] target row norm, size needs to be at least [i]
 * @param [in] norm norm type to be evaluated
 */
template <typename math_t>
void matrixRowNorm(const raft::handle_t& handle,
                   raft::device_csr_matrix_view<math_t, int, int, int> matrix,
                   math_t* target,
                   raft::linalg::NormType norm)
{
  auto csr_struct_in = matrix.structure_view();
  raft::sparse::linalg::rowNormCsr(handle,
                                   csr_struct_in.get_indptr().data(),
                                   matrix.get_elements().data(),
                                   csr_struct_in.get_nnz(),
                                   csr_struct_in.get_n_rows(),
                                   target,
                                   norm);
}

/**
 * @brief Extract CSR rows to dense
 *
 * Extraction of individual rows of a CSR matrix into a dense
 * array with column major order.
 *
 * @param [in] indptr row index pointer of CSR input [n_rows + 1]
 * @param [in] indices column indices of CSR input [nnz = indptr[nrows + 1]]
 * @param [in] data values of CSR input [nnz = indptr[nrows + 1]]
 * @param [in] n_rows number of matrix rows
 * @param [in] n_cols number of matrix columns
 * @param [out] output dense array, size needs to be at least [num_indices * n_cols]
 * @param [in] row_indices row indices to extract [num_indices]
 * @param [in] num_indices number of indices to extract
 * @param [in] stream cuda stream
 */
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
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

struct rowsize : public thrust::unary_function<int, int> {
  const int* indptr_;
  rowsize(const int* indptr) : indptr_(indptr) {}

  __device__ int64_t operator()(const int& x) const { return indptr_[x + 1] - indptr_[x]; }
};

/**
 * @brief Extract matrix rows to sub matrix
 *
 * This is the specialized version for
 *     'DENSE -> CSR (data owning)'
 *
 * Note: just added for compilation, should not be hit at runtime
 */
template <typename math_t, typename LayoutPolicyIn>
void extractRows(raft::device_matrix_view<math_t, int, LayoutPolicyIn> matrix_in,
                 raft::device_csr_matrix<math_t, int, int, int> matrix_out,
                 const int* row_indices,
                 int num_indices,
                 const raft::handle_t& handle)
{
  ASSERT(false, "extractRows from DENSE-CSR not implemented.");
}

/**
 * @brief Extract matrix rows to sub matrix
 *
 * This is the specialized version for
 *     'DENSE -> DENSE (raw pointer)'
 *
 * TODO: move this functionality to
 * https://github.com/rapidsai/raft/issues/1524
 *
 * @param [in] matrix_in matrix input (dense view)  [i, j]
 * @param [out] matrix_out matrix output raw pointer, size at least num_indices*j
 * @param [in] row_indices row indices to extract [num_indices]
 * @param [in] num_indices number of indices to extract
 * @param [in] handle raft handle
 */
template <typename math_t, typename LayoutPolicyIn>
void extractRows(raft::device_matrix_view<math_t, int, LayoutPolicyIn> matrix_in,
                 math_t* matrix_out,
                 const int* row_indices,
                 int num_indices,
                 const raft::handle_t& handle)
{
  ASSERT(matrix_in.stride(0) == 1, "Matrix needs to be column major");
  ASSERT(matrix_in.stride(1) == matrix_in.extent(0), "No padding supported");

  raft::matrix::copyRows<math_t, int, size_t>(matrix_in.data_handle(),
                                              matrix_in.extent(0),
                                              matrix_in.extent(1),
                                              matrix_out,
                                              row_indices,
                                              num_indices,
                                              handle.get_stream(),
                                              false);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Extract matrix rows to sub matrix
 *
 * This is the specialized version for
 *     'CSR -> DENSE (raw pointer)'
 *
 * @param [in] matrix_in matrix input in CSR  [i, j]
 * @param [out] matrix_out matrix output raw pointer, size at least num_indices*j
 * @param [in] row_indices row indices to extract [num_indices]
 * @param [in] num_indices number of indices to extract
 * @param [in] handle raft handle
 */
template <typename math_t>
void extractRows(raft::device_csr_matrix_view<math_t, int, int, int> matrix_in,
                 math_t* matrix_out,
                 const int* row_indices,
                 int num_indices,
                 const raft::handle_t& handle)
{
  auto stream        = handle.get_stream();
  auto csr_struct_in = matrix_in.structure_view();

  // initialize dense target
  thrust::device_ptr<math_t> output_ptr(matrix_out);
  thrust::fill(thrust::cuda::par.on(stream),
               output_ptr,
               output_ptr + num_indices * csr_struct_in.get_n_cols(),
               (math_t)0);

  int* indptr  = csr_struct_in.get_indptr().data();
  int* indices = csr_struct_in.get_indices().data();
  math_t* data = matrix_in.get_elements().data();

  // copy with 1 warp per row for now, blocksize 256
  const dim3 bs(32, 8, 1);
  const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
  extractDenseRowsFromCSR<math_t>
    <<<gs, bs, 0, stream>>>(matrix_out, indptr, indices, data, row_indices, num_indices);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

namespace {
int computeIndptrForSubset(
  int* indptr_in, int* indptr_out, const int* row_indices, int num_indices, cudaStream_t stream)
{
  thrust::device_ptr<int> row_sizes_ptr(indptr_out);
  thrust::device_ptr<const int> row_new_indices_ptr(row_indices);
  thrust::transform_inclusive_scan(thrust::cuda::par.on(stream),
                                   row_new_indices_ptr,
                                   row_new_indices_ptr + num_indices,
                                   row_sizes_ptr + 1,
                                   rowsize(indptr_in),
                                   thrust::plus<int>());

  // retrieve nnz from indptr_in[num_indices]
  int nnz;
  raft::update_host(&nnz, indptr_out + num_indices, 1, stream);
  cudaStreamSynchronize(stream);
  return nnz;
}

}  // namespace

/**
 * @brief copy row pointers from device to host
 *
 * This is only implemented for CSR
 *
 * @param [in] matrix matrix input in CSR  [i, j]
 * @param [out] host_indptr indptr in host  [i + 1]
 * @param [in] stream cuda stream
 */
template <typename math_t>
void copyIndptrToHost(raft::device_csr_matrix_view<math_t, int, int, int> matrix,
                      int* host_indptr,
                      cudaStream_t stream)
{
  raft::update_host(host_indptr,
                    matrix.structure_view().get_indptr().data(),
                    matrix.structure_view().get_n_rows() + 1,
                    stream);
  cudaStreamSynchronize(stream);
}

/**
 * @brief copy row pointers from device to host
 *
 * This is only implemented for CSR
 *
 * @param [in] matrix matrix input [i, j]
 * @param [out] host_indptr indptr in host  [i + 1]
 * @param [in] stream cuda stream
 */
template <typename math_t, typename LayoutPolicyIn>
void copyIndptrToHost(raft::device_matrix_view<math_t, int, LayoutPolicyIn> matrix,
                      int* host_indptr,
                      cudaStream_t stream)
{
  ASSERT(false, "Variant not implemented.");
}

/**
 * @brief Extract matrix rows to sub matrix
 *
 * This is the specialized version for
 *     'CSR -> CSR (data owning)'
 *
 * TODO: move this functionality to
 * https://github.com/rapidsai/raft/issues/1524
 *
 * @param [in] matrix_in matrix input in CSR  [i, j]
 * @param [out] matrix_out matrix output in CSR  [num_indices, j]
 * @param [in] row_indices row indices to extract [num_indices]
 * @param [in] num_indices number of indices to extract
 * @param [in] handle raft handle
 */
template <typename math_t>
void extractRows(raft::device_csr_matrix_view<math_t, int, int, int> matrix_in,
                 raft::device_csr_matrix<math_t, int, int, int>& matrix_out,
                 const int* row_indices,
                 int num_indices,
                 const raft::handle_t& handle)
{
  auto stream        = handle.get_stream();
  auto csr_struct_in = matrix_in.structure_view();
  int* indptr_in     = csr_struct_in.get_indptr().data();
  int* indices_in    = csr_struct_in.get_indices().data();
  math_t* data_in    = matrix_in.get_elements().data();

  auto csr_struct_out = matrix_out.structure_view();
  int* indptr_out     = csr_struct_out.get_indptr().data();

  int nnz = computeIndptrForSubset(indptr_in, indptr_out, row_indices, num_indices, stream);

  // this might invalidate indices/data pointers!
  matrix_out.initialize_sparsity(nnz);

  int* indices_out = matrix_out.structure_view().get_indices().data();
  math_t* data_out = matrix_out.get_elements().data();

  // copy with 1 warp per row for now, blocksize 256
  const dim3 bs(32, 8, 1);
  const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
  extractCSRRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(
    indptr_out, indices_out, data_out, indptr_in, indices_in, data_in, row_indices, num_indices);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Extract matrix rows to sub matrix
 *
 * This is the specialized version for
 *     'DENSE -> CSR (raw pointers)'
 *
 * Note: just added for compilation, should not be hit at runtime
 */
template <typename math_t, typename LayoutPolicyIn>
void extractRows(raft::device_matrix_view<math_t, int, LayoutPolicyIn> matrix_in,
                 int** indptr_out,
                 int** indices_out,
                 math_t** data_out,
                 int* nnz,
                 const int* row_indices,
                 int num_indices,
                 const raft::handle_t& handle)
{
  ASSERT(false, "extractRows not implemented for DENSE->CSR");
}

/**
 * @brief Extract matrix rows to sub matrix
 *
 * This is the specialized version for
 *     'CSR -> CSR (raw pointers)'
 *
 * Warning: this specialization will allocate the the required arrays in device memory.
 *
 * @param [in] matrix_in matrix input in CSR  [i, j]
 * @param [out] indptr_out row index pointer of CSR output [num_indices + 1]
 * @param [out] indices_out column indices of CSR output [nnz = indptr_out[num_indices + 1]]
 * @param [out] data_out values of CSR output [nnz = indptr_out[num_indices + 1]]
 * @param [out] nnz number of indices to extract
 * @param [in] row_indices row indices to extract [num_indices]
 * @param [in] num_indices number of indices to extract
 * @param [in] handle raft handle
 */
template <typename math_t>
void extractRows(raft::device_csr_matrix_view<math_t, int, int, int> matrix_in,
                 int** indptr_out,
                 int** indices_out,
                 math_t** data_out,
                 int* nnz,
                 const int* row_indices,
                 int num_indices,
                 const raft::handle_t& handle)
{
  auto stream        = handle.get_stream();
  auto csr_struct_in = matrix_in.structure_view();
  int* indptr_in     = csr_struct_in.get_indptr().data();
  int* indices_in    = csr_struct_in.get_indices().data();
  math_t* data_in    = matrix_in.get_elements().data();

  // allocate indptr
  auto* rmm_alloc = rmm::mr::get_current_device_resource();
  *indptr_out     = (int*)rmm_alloc->allocate((num_indices + 1) * sizeof(int), stream);

  *nnz = computeIndptrForSubset(indptr_in, *indptr_out, row_indices, num_indices, stream);

  // allocate indices, data
  *indices_out = (int*)rmm_alloc->allocate(*nnz * sizeof(int), stream);
  *data_out    = (math_t*)rmm_alloc->allocate(*nnz * sizeof(math_t), stream);

  // copy with 1 warp per row for now, blocksize 256
  const dim3 bs(32, 8, 1);
  const dim3 gs(raft::ceildiv(num_indices, (int)bs.y), 1, 1);
  extractCSRRowsFromCSR<math_t><<<gs, bs, 0, stream>>>(
    *indptr_out, *indices_out, *data_out, indptr_in, indices_in, data_in, row_indices, num_indices);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace SVM
}  // namespace ML
