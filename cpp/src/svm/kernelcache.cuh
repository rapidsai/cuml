/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "sparse_util.cuh"
#include <cuml/svm/svm_parameter.h>
#include <raft/core/handle.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/linalg/init.cuh>
#include <raft/util/cache.cuh>
#include <raft/util/cache_util.cuh>

#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reverse.h>

#include <cuml/common/logger.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstddef>

namespace ML {
namespace SVM {

namespace {  // unnamed namespace to avoid multiple definition error
/**
 * @brief Calculate mapping from working set to kernel tile columns
 *
 * On exit, out[k] is defined so that unique[out[k]] == ws[k] % n_rows.
 *
 * @param [in] ws array with working set indices, size [n_ws]
 * @param [in] n_ws number of elements in the working set
 * @param [in] n_rows number of rows in the original problem
 * @param [in] unique array with indices of unique vectors in the working set,
 *     size [n_unique]
 * @param [in] n_unique number of elements in the unique array
 * @param [out] out array with workspace idx to column idx mapping, size [n_ws]
 */
__global__ void mapColumnIndices(
  const int* ws, int n_ws, int n_rows, const int* unique, int n_unique, int* out)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_ws) {
    int idx = ws[tid] % n_rows;
    int k   = 0;
    // we have only max 1024 elements, we do a linear search
    for (int i = 0; i < n_unique; i++) {
      if (unique[i] == idx) k = i;
      // since the unique array is derived from ws, the search will always return
      // the correct idx where unique[k] == idx
    }
    out[tid] = k;
  }
}

/**
 * @brief Re-raise working set indexes to dual space
 *
 * On exit, out is the permutation of n_ws such that out[k]%n_rows == n_ws_perm[k]
 * In case n_ws_perm contains duplicates they are considered to
 * represent primal first and dual second
 *
 * @param [in] ws array with working set indices, size [n_ws]
 * @param [in] n_ws number of elements in the working set
 * @param [in] n_rows number of rows in the original problem
 * @param [in] n_ws_perm array with indices of vectors in the working set, size [n_ws]
 * @param [out] out array with workspace idx to column idx mapping, size [n_ws]
 */
__global__ void mapColumnIndicesToDualSpace(
  const int* ws, int n_ws, int n_rows, const int* n_ws_perm, int* out)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_ws) {
    int wsx      = ws[tid];
    int idx      = wsx % n_rows;
    bool is_dual = idx < wsx;
    int k        = -1;
    // we have only max 1024 elements, we do a linear search
    for (int i = 0; i < n_ws; i++) {
      if (n_ws_perm[i] == idx && (is_dual || k < 0)) k = i;
      // since the array is derived from ws, the search will always return
      //  a) the first occurrence k for primal idx
      //  b) the last occurrence k for dual idx
    }
    out[k] = wsx;
  }
}

template <typename math_t>
struct select_at_index : public thrust::unary_function<int, math_t> {
  const math_t* dot_;
  select_at_index(const math_t* dot) : dot_(dot) {}

  __device__ math_t operator()(const int& i) const { return dot_[i]; }
};

template <typename math_t>
class BatchCache {
 public:
  BatchCache(int n_vec, float cache_size = 200) : n_vec(n_vec), cache_size(cache_size) {}

  ~BatchCache()
  {
    for (auto* cache : caches)
      delete cache;
  }

  void Initialize(cudaStream_t stream, int batch_size_base, int* dummy, int n_ws)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(dummy, 0, n_ws * 2 * sizeof(int), stream));
    for (int offset = 0; offset < n_vec; offset += batch_size_base) {
      int batch_size         = std::min(batch_size_base, n_vec - offset);
      float cache_size_batch = cache_size * batch_size / n_vec;
      auto* cache            = new raft::cache::Cache<math_t>(stream, batch_size, cache_size_batch);

      // initialize internal data structures
      if (cache->GetSize() > 0) {
        int cached;
        cache->GetCacheIdxPartitioned(dummy, n_ws, dummy + n_ws, &cached, stream);
      }

      caches.push_back(cache);
    }

    int size = GetSize();
    for (auto* cache : caches) {
      ASSERT(cache->GetSize() == size, "Unexpected cache size");
    }
  }

  int GetSize(int batch_idx = 0) const
  {
    return caches.size() > 0 ? caches[batch_idx]->GetSize() : 0;
  }

  void GetCacheIdxPartitionedStable(
    int* keys, int n, int* cache_idx, int* n_cached, cudaStream_t stream)
  {
    caches[0]->GetCacheIdxPartitioned(keys, n, cache_idx, n_cached, stream);

    // reverse the uncached values (due to cub::DevicePartition:Flagged)
    int n_uncached = n - *n_cached;
    if (n_uncached > 0) {
      thrust::device_ptr<int> keys_v(keys + *n_cached);
      thrust::reverse(thrust::cuda::par.on(stream), keys_v, keys_v + n_uncached);
      thrust::device_ptr<int> cache_idx_v(cache_idx + *n_cached);
      thrust::reverse(thrust::cuda::par.on(stream), cache_idx_v, cache_idx_v + n_uncached);
    }
  }

  void GetVecs(int batch_idx, const int* idx, int n, math_t* out, cudaStream_t stream)
  {
    caches[batch_idx]->GetVecs(idx, n, out, stream);
  }

  void AssignAndStoreVecs(int batch_idx,
                          int* keys,
                          int n,
                          int* cache_idx,
                          const math_t* tile,
                          int n_tile,
                          cudaStream_t stream)
  {
    // here we assume that the input keys are already ordered by cache_idx
    // this will prevent AssignCacheIdx to modify it further
    caches[batch_idx]->AssignCacheIdx(keys, n, cache_idx, stream);
    caches[batch_idx]->StoreVecs(tile, n_tile, n, cache_idx, stream);
  }

 private:
  int n_vec;         //!< Total number of elements (sum over all caches)
  float cache_size;  //!< total over all caches in MiB
  std::vector<raft::cache::Cache<math_t>*> caches;
};

}  // end unnamed namespace

// Strategy:
//    - rows && cols sufficiently small to allow n_ws * n_rows/n_cols to fit in memory
//       * extract dense rows for dot product
//       * No batching of update step
//
//    if n_rows too large:
//       -> Enable batching in update step to limit kernel_big memory
//
//    if n_cols too large: (CSR only support)
//       -> extract CSR rows instead of dense for kernel compute
//
template <typename math_t>
class KernelCache {
 public:
  KernelCache(const raft::handle_t& handle,
              const MLCommon::Matrix::Matrix<math_t>& matrix,
              int n_rows,
              int n_cols,
              int n_ws,
              raft::distance::kernels::GramMatrixBase<math_t>* kernel,
              raft::distance::kernels::KernelType kernel_type,
              float cache_size = 200,
              SvmType svmType  = C_SVC)
    : batch_cache(n_rows, cache_size),
      handle(handle),
      kernel(kernel),
      kernel_type(kernel_type),
      matrix(matrix),
      n_rows(n_rows),
      n_cols(n_cols),
      n_ws(n_ws),
      svmType(svmType),
      kernel_tile(0, handle.get_stream()),
      matrix_l2(0, handle.get_stream()),
      matrix_l2_ws(0, handle.get_stream()),
      ws_idx_mod(n_ws, handle.get_stream()),
      ws_idx_mod_dual(svmType == EPSILON_SVR ? n_ws : 0, handle.get_stream()),
      x_ws_csr(handle, 0, 0, 0),
      x_ws_dense(handle, 0, 0, 0),
      indptr_batched(0, handle.get_stream()),
      d_temp_storage(0, handle.get_stream()),
      ws_cache_idx(n_ws * 2, handle.get_stream())
  {
    ASSERT(kernel != nullptr, "Kernel pointer required for KernelCache!");
    stream = handle.get_stream();

    // FIXME: not sure if this has to be static
    batching_enabled = false;
    is_csr           = !matrix.is_dense();
    sparse_extract   = false;
    batch_size_base  = n_rows;

    // enable batching for kernel > 1 GB
    size_t big_kernel_max_bytes = 1 << 30;
    if (n_rows * n_ws * sizeof(math_t) > big_kernel_max_bytes) {
      batching_enabled = true;
      // only select based on desired big-kernel size
      batch_size_base = big_kernel_max_bytes / n_ws / sizeof(math_t);
    }

    batch_cache.Initialize(stream, batch_size_base, ws_cache_idx.data(), n_ws);
    kernel_tile.reserve(n_ws * std::max(batch_size_base, n_ws), stream);

    // enable sparse row extraction for sparse input where n_ws * n_cols > 1 GB
    // Warning: kernel computation will be much slower!
    size_t extract_rows_max_bytes = 1 << 30;
    if (is_csr && (n_cols * n_ws * sizeof(math_t) > extract_rows_max_bytes)) {
      sparse_extract = true;
    }

    if (sparse_extract)
      x_ws_matrix = &x_ws_csr;
    else
      x_ws_matrix = &x_ws_dense;

    // store matrix l2 norm for RBF kernels
    if (kernel_type == raft::distance::kernels::KernelType::RBF) {
      matrix_l2.reserve(n_rows, stream);
      matrix_l2_ws.reserve(n_ws, stream);
      ML::SVM::matrixRowNorm(handle, matrix, matrix_l2.data(), raft::linalg::NormType::L2Norm);
    }

    // additional row pointer information needed for batched CSR access
    // copy matrix row pointer to host to compute partial nnz on the fly
    if (is_csr && batching_enabled) {
      host_indptr.reserve(n_rows + 1);
      indptr_batched.reserve(batch_size_base + 1, stream);
      raft::update_host(host_indptr.data(), matrix.as_csr()->get_indptr(), n_rows + 1, stream);
    }

    // Init cub buffers
    cub::DeviceRadixSort::SortKeys(NULL,
                                   d_temp_storage_size,
                                   ws_idx_mod.data(),
                                   ws_idx_mod.data(),
                                   n_ws,
                                   0,
                                   sizeof(int) * 8,
                                   stream);
    d_temp_storage.resize(d_temp_storage_size, stream);
  }
  ~KernelCache(){};

  struct BatchDescriptor {
    int batch_id;
    int offset;
    int batch_size;
    math_t* kernel_data;
    int* nz_da_idx;
    int nnz_da;
    int n_cached;
  };

  enum CacheState {
    READY                = 0,
    WS_INITIALIZED       = 1,
    BATCHING_INITIALIZED = 2,
  };

  // reorder indices to cached/uncached
  void InitWorkingSet(const int* ws_idx)
  {
    ASSERT(cache_state != CacheState::WS_INITIALIZED, "Working set has already been initialized!");
    ASSERT(cache_state != CacheState::BATCHING_INITIALIZED, "Previous batching step incomplete!");
    this->ws_idx = ws_idx;
    if (svmType == EPSILON_SVR) {
      GetVecIndices(ws_idx, n_ws, ws_idx_mod.data());
    } else {
      raft::copy(ws_idx_mod.data(), ws_idx, n_ws, stream);
    }

    // perform reordering of indices to partition into cached/uncached
    // batch_id 0 should behave the same as all other batches
    if (batch_cache.GetSize() > 0) {
      int n_cached = 0;
      batch_cache.GetCacheIdxPartitionedStable(
        ws_idx_mod.data(), n_ws, ws_cache_idx.data(), &n_cached, stream);
      int n_uncached = n_ws - n_cached;
      if (n_uncached > 0) {
        // we also need to make sure that the next cache assignment does not need to rearrange!
        // this way the resulting ws_idx_mod keys won't change during the cache update
        int* tmp = (int*)kernel_tile.data();
        cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                        d_temp_storage_size,
                                        ws_cache_idx.data() + n_cached,
                                        tmp,
                                        ws_idx_mod.data() + n_cached,
                                        tmp + n_ws,
                                        n_uncached,
                                        0,
                                        sizeof(int) * 8,
                                        stream);

        // raft::copy(ws_cache_idx.data() + n_cached, tmp, n_uncached, stream); // TODO: debug,not
        // really needed
        raft::copy(ws_idx_mod.data() + n_cached, tmp + n_ws, n_uncached, stream);
      }
    }

    // re-compute original (dual) indices that got flattened by GetVecIndices
    if (svmType == EPSILON_SVR) {
      mapColumnIndicesToDualSpace<<<raft::ceildiv(n_ws, TPB), TPB, 0, stream>>>(
        ws_idx, n_ws, n_rows, ws_idx_mod.data(), ws_idx_mod_dual.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    cache_state = CacheState::WS_INITIALIZED;
  }

  // needs call to 'InitWorkingSet' first
  const int* getKernelIndices(int allow_dual)
  {
    ASSERT(cache_state != CacheState::READY, "Working set not initialized!");
    if (allow_dual && svmType == EPSILON_SVR) {
      return ws_idx_mod_dual.data();
    } else {
      return ws_idx_mod.data();
    }
  }

  // needs call to 'InitWorkingSet' first
  // does currently not utilize cache BUT honors order of cached/uncached partition
  // maybe we can still utilize existing cache entries but not update it
  math_t* getSquareTileWithoutCaching()
  {
    ASSERT(cache_state != CacheState::READY, "Working set not initialized!");
    ASSERT(cache_state != CacheState::BATCHING_INITIALIZED, "Previous batching step incomplete!");

    ML::SVM::extractRows<math_t>(matrix, *x_ws_matrix, ws_idx_mod.data(), n_ws, handle);

    // extract dot array for RBF
    if (kernel_type == raft::distance::kernels::KernelType::RBF) {
      selectValueSubset(matrix_l2_ws.data(), matrix_l2.data(), ws_idx_mod.data(), n_ws);
    }

    // compute kernel
    {
      MLCommon::Matrix::DenseMatrix<math_t> kernel_matrix(kernel_tile.data(), n_ws, n_ws);
      KernelOp(handle,
               kernel,
               *x_ws_matrix,
               *x_ws_matrix,
               kernel_matrix,
               matrix_l2_ws.data(),
               matrix_l2_ws.data());
    }
    return kernel_tile.data();
  }

  // input should only be in range of [0, nvec) (primal space)
  BatchDescriptor InitFullTileBatching(int* nz_da_idx, int nnz_da)
  {
    ASSERT(cache_state != CacheState::READY, "Working set not initialized!");
    ASSERT(cache_state != CacheState::BATCHING_INITIALIZED, "Previous batching step incomplete!");

    int n_cached = 0;
    if (batch_cache.GetSize() > 0) {
      // we only do this once per workingset for cache 0 - all others should behave the same
      batch_cache.GetCacheIdxPartitionedStable(
        nz_da_idx, nnz_da, ws_cache_idx.data() + n_ws, &n_cached, stream);
    }

    int n_uncached = nnz_da - n_cached;
    if (n_uncached > 0) {
      ML::SVM::extractRows<math_t>(matrix, *x_ws_matrix, nz_da_idx + n_cached, n_uncached, handle);
      // extract dot array for RBF
      if (kernel_type == raft::distance::kernels::KernelType::RBF) {
        selectValueSubset(matrix_l2_ws.data(), matrix_l2.data(), nz_da_idx + n_cached, n_uncached);
      }
    }

    cache_state = CacheState::BATCHING_INITIALIZED;

    return {.batch_id    = -1,
            .offset      = 0,
            .batch_size  = 0,
            .kernel_data = nullptr,
            .nz_da_idx   = nz_da_idx,
            .nnz_da      = nnz_da,
            .n_cached    = n_cached};
  }

  // Assumption:
  // 1. cached lines are identical for all batches !
  //     -- this is needed to re-use x_ws_matrix
  // 2. permutation is identical for all batches & is stored
  // 3. permutation can be used to reorder nz_da array for multiply
  bool getNextBatchKernel(BatchDescriptor& batch_descriptor)
  {
    ASSERT(cache_state == CacheState::BATCHING_INITIALIZED, "Batching step not initialized!");

    int offset = batch_descriptor.offset + batch_descriptor.batch_size;
    if (offset >= n_rows) {
      cache_state = CacheState::READY;
      return false;
    }

    int batch_size = std::min(batch_size_base, n_rows - offset);
    int batch_id   = offset / batch_size_base;

    ASSERT(offset % batch_size_base == 0, "Inconsistent offset!");
    ASSERT(batch_id == batch_descriptor.batch_id + 1, "Inconsistent batch_id!");

    int nnz_da     = batch_descriptor.nnz_da;
    int n_cached   = batch_descriptor.n_cached;
    int n_uncached = nnz_da - n_cached;
    if (batch_cache.GetSize() > 0) {
      // re-init ws_cache_idx for each batch
      raft::copy(ws_cache_idx.data(), ws_cache_idx.data() + n_ws, nnz_da, stream);
    }

    // fill in n_cached ids from cache
    if (n_cached > 0) {
      batch_cache.GetVecs(batch_id, ws_cache_idx.data(), n_cached, kernel_tile.data(), stream);
    }

    if (n_uncached > 0) {
      int* ws_idx_new  = batch_descriptor.nz_da_idx + n_cached;
      math_t* tile_new = kernel_tile.data() + (size_t)n_cached * batch_size;

      const MLCommon::Matrix::Matrix<math_t>* batch_matrix = nullptr;
      if (batch_size == n_rows) {
        batch_matrix = &matrix;
      } else if (matrix.is_dense()) {
        auto dense_matrix = matrix.as_dense();
        batch_matrix      = new MLCommon::Matrix::DenseMatrix(
          dense_matrix->get_data() + offset, batch_size, n_cols, false, n_rows);
      } else {
        // create indptr array for batch interval
        // indptr_batched = indices[offset, offset+batch_size+1] - indices[offset]
        auto csr_matrix = matrix.as_csr();
        int batch_nnz   = host_indptr[offset + batch_size] - host_indptr[offset];
        {
          thrust::device_ptr<int> inptr_src(csr_matrix->get_indptr() + offset);
          thrust::device_ptr<int> inptr_tgt(indptr_batched.data());
          thrust::transform(thrust::cuda::par.on(stream),
                            inptr_src,
                            inptr_src + batch_size + 1,
                            thrust::make_constant_iterator(host_indptr[offset]),
                            inptr_tgt,
                            thrust::minus<int>());
        }
        batch_matrix =
          new MLCommon::Matrix::CsrMatrix(indptr_batched.data(),
                                          csr_matrix->get_indices() + host_indptr[offset],
                                          csr_matrix->get_data() + host_indptr[offset],
                                          batch_nnz,
                                          batch_size,
                                          n_cols);
      }

      // compute kernel
      MLCommon::Matrix::DenseMatrix<math_t> kernel_matrix(tile_new, batch_size, n_uncached);
      KernelOp(handle,
               kernel,
               *batch_matrix,
               *x_ws_matrix,
               kernel_matrix,
               matrix_l2.data() + offset,
               matrix_l2_ws.data());

      RAFT_CUDA_TRY(cudaPeekAtLastError());

      if (batch_matrix != &matrix) delete batch_matrix;

      if (batch_cache.GetSize() > 0) {
        // AssignCacheIdx should not permute ws_idx_new anymore as we have sorted
        // it already during InitWorkingSet
        batch_cache.AssignAndStoreVecs(batch_id,
                                       ws_idx_new,
                                       n_uncached,
                                       ws_cache_idx.data() + n_cached,
                                       tile_new,
                                       batch_size,
                                       stream);
      }
    }

    batch_descriptor.batch_id    = batch_id;
    batch_descriptor.offset      = offset;
    batch_descriptor.batch_size  = batch_size;
    batch_descriptor.kernel_data = kernel_tile.data();

    return true;
  }

  void selectValueSubset(math_t* target, const math_t* source, const int* indices, int num_indices)
  {
    thrust::device_ptr<const int> indices_ptr(indices);
    thrust::device_ptr<math_t> target_ptr(target);
    thrust::transform(thrust::cuda::par.on(stream),
                      indices_ptr,
                      indices_ptr + num_indices,
                      target_ptr,
                      select_at_index(source));
  }

  /** @brief Get the original training vector idx.
   *
   * Only used for SVR (for SVC this is identity operation).
   *
   * For SVR we have duplicate set of training vectors, we return the original
   * idx, which is simply ws_idx % n_rows.
   *
   * @param [in] ws_idx array of working set indices, size [n_ws]
   * @param [in] n_ws number of elements in the working set
   * @param [out] vec_idx original training vector indices, size [n_ws]
   */
  void GetVecIndices(const int* ws_idx, int n_ws, int* vec_idx)
  {
    int n = n_rows;
    raft::linalg::unaryOp(
      vec_idx, ws_idx, n_ws, [n] __device__(math_t y) { return y < n ? y : y - n; }, stream);
  }

 private:
  const MLCommon::Matrix::Matrix<math_t>& matrix;

  const int* ws_idx;  //!< ptr to the original working set

  bool batching_enabled;
  bool is_csr;
  bool sparse_extract;
  int batch_size_base;

  // cache state
  CacheState cache_state = CacheState::READY;

  rmm::device_uvector<math_t> kernel_tile;

  // permutation of working set indices to partition cached/uncached
  rmm::device_uvector<int> ws_idx_mod;
  rmm::device_uvector<int> ws_idx_mod_dual;

  // tmp storage for row extractions
  MLCommon::Matrix::CsrMatrix<math_t> x_ws_csr;
  MLCommon::Matrix::DenseMatrix<math_t> x_ws_dense;
  MLCommon::Matrix::Matrix<math_t>* x_ws_matrix = nullptr;

  // matrix l2 norm for RBF kernels
  rmm::device_uvector<math_t> matrix_l2;
  rmm::device_uvector<math_t> matrix_l2_ws;

  // additional row pointer information needed for batched CSR access
  // copy matrix row pointer to host to compute partial nnz on the fly
  std::vector<int> host_indptr;
  rmm::device_uvector<int> indptr_batched;

  raft::distance::kernels::GramMatrixBase<math_t>* kernel;
  raft::distance::kernels::KernelType kernel_type;

  int n_rows;  //!< number of rows in x
  int n_cols;  //!< number of columns in x
  int n_ws;    //!< number of elements in the working set

  /// cache position of a workspace vectors
  rmm::device_uvector<int> ws_cache_idx;

  // tmp storage for cub
  rmm::device_uvector<char> d_temp_storage;
  size_t d_temp_storage_size = 0;

  const raft::handle_t handle;

  BatchCache<math_t> batch_cache;

  cudaStream_t stream;
  SvmType svmType;

  const int TPB = 256;  //!< threads per block for kernels launched
};

/**
 * @brief Buffer to store a kernel tile
 *
 * We calculate the kernel matrix for the vectors in the working set.
 * For every vector x_i in the working set, we always calculate a full row of the
 * kernel matrix K(x_j, x_i), j=1..n_rows.
 *
 * A kernel tile stores all the kernel rows for the working set, i.e. K(x_j, x_i)
 * for all i in the working set, and j in 1..n_rows. For details about the kernel
 * tile layout, see KernelCacheOld::GetTile
 *
 * The kernel values can be cached to avoid repeated calculation of the kernel
 * function.
 */
template <typename math_t>
class KernelCacheOld {
 public:
  /**
   * Construct an object to manage kernel cache
   *
   * @param handle reference to raft::handle_t implementation
   * @param x device array of training vectors in column major format,
   *   size [n_rows x n_cols]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param n_ws size of working set
   * @param kernel pointer to kernel (default linear)
   * @param cache_size (default 200 MiB)
   * @param svmType is this SVR or SVC
   */
  KernelCacheOld(const raft::handle_t& handle,
                 const math_t* x,
                 int n_rows,
                 int n_cols,
                 int n_ws,
                 raft::distance::kernels::GramMatrixBase<math_t>* kernel,
                 float cache_size = 200,
                 SvmType svmType  = C_SVC)
    : cache(handle.get_stream(), n_rows, cache_size),
      handle(handle),
      kernel(kernel),
      x(x),
      n_rows(n_rows),
      n_cols(n_cols),
      n_ws(n_ws),
      svmType(svmType),
      cublas_handle(handle.get_cublas_handle()),
      d_num_selected_out(handle.get_stream()),
      d_temp_storage(0, handle.get_stream()),
      x_ws(0, handle.get_stream()),
      tile(0, handle.get_stream()),
      unique_idx(n_ws, handle.get_stream()),
      k_col_idx(n_ws, handle.get_stream()),
      ws_cache_idx(n_ws, handle.get_stream())
  {
    ASSERT(kernel != nullptr, "Kernel pointer required for KernelCacheOld!");
    stream = handle.get_stream();

    size_t kernel_tile_size = (size_t)n_ws * n_rows;
    CUML_LOG_DEBUG("Allocating kernel tile, size: %zu MiB",
                   kernel_tile_size * sizeof(math_t) / (1024 * 1024));
    tile.resize(kernel_tile_size, handle.get_stream());

    size_t x_ws_tile_size = (size_t)n_ws * n_cols;
    CUML_LOG_DEBUG("Allocating x_ws, size: %zu KiB", x_ws_tile_size / (1024));
    x_ws.resize(x_ws_tile_size, handle.get_stream());

    // Default kernel_column_idx map for SVC
    raft::linalg::range(k_col_idx.data(), n_ws, stream);

    // Init cub buffers
    std::size_t bytes1{};
    std::size_t bytes2{};
    cub::DeviceRadixSort::SortKeys(
      NULL, bytes1, unique_idx.data(), unique_idx.data(), n_ws, 0, sizeof(int) * 8, stream);
    cub::DeviceSelect::Unique(
      NULL, bytes2, unique_idx.data(), unique_idx.data(), d_num_selected_out.data(), n_ws, stream);
    d_temp_storage_size = std::max(bytes1, bytes2);
    d_temp_storage.resize(d_temp_storage_size, stream);
  }

  ~KernelCacheOld(){};

  /**
   * @brief Get all the kernel matrix rows for the working set.
   *
   * The kernel matrix is stored in column major format:
   * kernel[row_id, col_id] = kernel[row_id + col_id * n_rows]
   *
   * For SVC:
   * kernel is rectangular with size [n_rows * n_ws], so:
   * \f[ row_id \in [0..n_rows-1], col_id \in [0..n_ws-1] \f]
   *
   * The columns correspond to the vectors in the working set. For example:
   * Let's assume thet the working set are vectors x_5, x_9, x_0, and x_4,
   * then ws_idx = [5, 9, 0 ,4]. The second column of the kernel matrix,
   * kernel[i + 1*n_rows] (\f[ i \in [0..n_rows-1] \f]), stores the kernel
   * matrix values for the second vector in the working set: K(x_i, x_9).
   *
   * For SVR:
   * We doubled the set of training vector, assigning:
   * \f[ x_{n_rows+i} = x_i for i \in [0..n_rows-1]. \f]
   *
   * The kernel matrix values are the same for x_i and x_{i+n_rows}, therefore
   * we store only kernel[row_id, col_id] for \f[ row_id \in [0..n_rows]. \f]
   *
   * Similarly, it can happen that two elements in the working set have the
   * same x vector. For example, if n_rows=10, then for ws = [5, 19, 15 0], the
   * first and third vectors are the same (x_5==x_15), therefore the
   * corresponding. columns in the kernel tile would be identical. We do not
   * store these duplicate columns for the kernel matrix. The size of the
   * kernel matrix is [n_rows * n_unique], where n_unique = 3 for our example.
   *
   * We map the working set indices to unique column indices using the k_col_idx
   * array. For the above example:  k_col_idx = [0, 1, 0, 2], i.e. the third vec
   * in the working set (x_15) is stored at column zero in the kernel matrix:
   * e.g.: K(x_i, x_15) = kernel[i + n_rows * 0]
   *
   * The returned kernel tile array allocated/deallocated by KernelCacheOld.
   *
   * Note: when cache_size > 0, the workspace indices can be permuted during
   * the call to GetTile. Use KernelCacheOld::GetWsIndices to query the actual
   * list of workspace indices.
   *
   * @param [in] ws_idx indices of the working set
   * @return pointer to the kernel tile [ n_rows x n_unique] K(x_j, x_q)
   */
  math_t* GetTile(const int* ws_idx)
  {
    this->ws_idx = ws_idx;
    GetUniqueIndices(ws_idx, n_ws, unique_idx.data(), &n_unique);
    if (cache.GetSize() > 0) {
      int n_cached;
      cache.GetCacheIdxPartitioned(
        unique_idx.data(), n_unique, ws_cache_idx.data(), &n_cached, stream);
      // collect already cached values
      cache.GetVecs(ws_cache_idx.data(), n_cached, tile.data(), stream);
      int non_cached = n_unique - n_cached;
      if (non_cached > 0) {
        int* ws_idx_new = unique_idx.data() + n_cached;
        // AssignCacheIdx can permute ws_idx_new, therefore it has to come
        // before calcKernel. Could come on separate stream to do collectrows
        // while AssignCacheIdx runs
        cache.AssignCacheIdx(ws_idx_new,
                             non_cached,
                             ws_cache_idx.data() + n_cached,
                             stream);  // cache stream

        // collect training vectors for kernel elements that needs to be calculated
        raft::matrix::copyRows<math_t, int, size_t>(
          x, n_rows, n_cols, x_ws.data(), ws_idx_new, non_cached, stream, false);
        math_t* tile_new = tile.data() + (size_t)n_cached * n_rows;

        MLCommon::Matrix::DenseMatrix<math_t> x_mat(const_cast<math_t*>(x), n_rows, n_cols);
        MLCommon::Matrix::DenseMatrix<math_t> x_ws_mat(x_ws.data(), non_cached, n_cols);
        MLCommon::Matrix::DenseMatrix<math_t> kernel_mat(tile_new, n_rows, non_cached);

        KernelOp(handle, kernel, x_mat, x_ws_mat, kernel_mat);
        //(*kernel)(x_mat, x_ws_mat, kernel_mat, handle);
        // We need AssignCacheIdx to be finished before calling StoreCols
        cache.StoreVecs(tile_new, n_rows, non_cached, ws_cache_idx.data() + n_cached, stream);
      }
    } else {
      if (n_unique > 0) {
        // collect all the feature vectors in the working set
        raft::matrix::copyRows<math_t, int, size_t>(
          x, n_rows, n_cols, x_ws.data(), unique_idx.data(), n_unique, stream, false);

        MLCommon::Matrix::DenseMatrix<math_t> x_mat(const_cast<math_t*>(x), n_rows, n_cols);
        MLCommon::Matrix::DenseMatrix<math_t> x_ws_mat(x_ws.data(), n_unique, n_cols);
        MLCommon::Matrix::DenseMatrix<math_t> kernel_mat(tile.data(), n_rows, n_unique);

        KernelOp(handle, kernel, x_mat, x_ws_mat, kernel_mat);
        //(*kernel)(x_mat, x_ws_mat, kernel_mat, handle);
      }
    }
    return tile.data();
  }

  /** Map workspace indices to kernel matrix indices.
   *
   * The kernel matrix is matrix of K[i+j*n_rows] = K(x_i, x_j), where
   * \f[ i \in [0..n_rows-1], and j=[0..n_unique-1] \f]
   *
   * The SmoBlockSolver needs to know where to find the kernel values that
   * correspond to vectors in the working set. Vector ws[i] corresponds to column
   * GetIdxMap()[i] in the kernel matrix.
   *
   * For SVC: GetIdxMap() == [0, 1, 2, ..., n_ws-1].
   *
   * SVR Example: n_rows = 3, n_train = 6, n_ws=4, ws_idx = [5 0 2 3]
   * Note that we have only two unique x vector in the training set:
   * ws_idx % n_rows = [2 0 2 0]
   *
   * To avoid redundant calculations, we just calculate the kernel values for the
   * unique elements from the working set: unique_idx = [0 2] , n_unique = 2, so
   * GetIdxMap() == [1 0 1 0].
   *
   * @return device array of index map size [n_ws], the array is owned by
   *   KernelCacheOld
   */
  int* GetColIdxMap()
  {
    if (svmType == EPSILON_SVR) {
      mapColumnIndices<<<raft::ceildiv(n_ws, TPB), TPB, 0, stream>>>(
        ws_idx, n_ws, n_rows, unique_idx.data(), n_unique, k_col_idx.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    // else k_col_idx is already initialized properly
    return k_col_idx.data();
  }

  /**
   * @brief Return the number of unique elements in the working set.
   *
   *  This is equal with the number of columns in the kernel tile.
   */
  int GetUniqueSize() { return n_unique; }

  const int* GetWsIndices()
  {
    if (svmType == C_SVC) {
      // the set if working set indices which were copied into unique_idx,
      // and permuted by the cache functions. These are trivially mapped
      // to the columns of the kernel tile.
      return unique_idx.data();
    } else {  // EPSILON_SVR
      // return the original working set elements. These are mapped to the
      // kernel tile columns by GetColIdxMap()
      return ws_idx;
    }
  }
  /** @brief Get the original training vector idx.
   *
   * Only used for SVR (for SVC this is identity operation).
   *
   * For SVR we have duplicate set of training vectors, we return the original
   * idx, which is simply ws_idx % n_rows.
   *
   * @param [in] ws_idx array of working set indices, size [n_ws]
   * @param [in] n_ws number of elements in the working set
   * @param [out] vec_idx original training vector indices, size [n_ws]
   */
  void GetVecIndices(const int* ws_idx, int n_ws, int* vec_idx)
  {
    int n = n_rows;
    raft::linalg::unaryOp(
      vec_idx, ws_idx, n_ws, [n] __device__(math_t y) { return y < n ? y : y - n; }, stream);
  }

 private:
  const math_t* x;    //!< pointer to the training vectors
  const int* ws_idx;  //!< pointer to the working set indices

  /// feature vectors in the current working set
  rmm::device_uvector<math_t> x_ws;
  /// cache position of a workspace vectors
  rmm::device_uvector<int> ws_cache_idx;

  rmm::device_uvector<math_t> tile;  //!< Kernel matrix  tile

  int n_rows;                        //!< number of rows in x
  int n_cols;                        //!< number of columns in x
  int n_ws;                          //!< number of elements in the working set
  int n_unique;                      //!< number of unique x vectors in the working set

  cublasHandle_t cublas_handle;

  raft::distance::kernels::GramMatrixBase<math_t>* kernel;

  const raft::handle_t handle;

  const int TPB = 256;  //!< threads per block for kernels launched

  raft::cache::Cache<math_t> cache;

  cudaStream_t stream;
  SvmType svmType;
  rmm::device_uvector<int> unique_idx;  //!< Training vector indices
  /// Column index map for the kernel tile
  rmm::device_uvector<int> k_col_idx;

  // Helper arrays for cub
  rmm::device_scalar<int> d_num_selected_out;
  rmm::device_uvector<char> d_temp_storage;
  size_t d_temp_storage_size = 0;

  /** Remove duplicate indices from the working set.
   *
   * The unique indices from the working set are stored in array unique_idx.
   * (For SVC this is just a copy of the working set. )
   *
   * @param [in] ws_idx device array of working set indices, size [n_ws]
   * @param [in] n_ws number of elements in the working set
   * @param [out] n_unique unique elements in the working set
   */
  void GetUniqueIndices(const int* ws_idx, int n_ws, int* unique_idx, int* n_unique)
  {
    if (svmType == C_SVC) {
      *n_unique = n_ws;
      raft::copy(unique_idx, ws_idx, n_ws, stream);
      return;
    }
    // for EPSILON_SVR
    GetVecIndices(ws_idx, n_ws, unique_idx);
    cub::DeviceRadixSort::SortKeys(d_temp_storage.data(),
                                   d_temp_storage_size,
                                   unique_idx,
                                   ws_cache_idx.data(),
                                   n_ws,
                                   0,
                                   sizeof(int) * 8,
                                   stream);
    cub::DeviceSelect::Unique(d_temp_storage.data(),
                              d_temp_storage_size,
                              ws_cache_idx.data(),
                              unique_idx,
                              d_num_selected_out.data(),
                              n_ws,
                              stream);
    raft::update_host(n_unique, d_num_selected_out.data(), 1, stream);
    handle.sync_stream(stream);
  }
};

};  // end namespace SVM
};  // end namespace ML
