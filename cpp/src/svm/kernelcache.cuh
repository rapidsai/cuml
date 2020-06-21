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

#include <common/cudart_utils.h>
#include <cuml/svm/svm_parameter.h>
#include <linalg/init.h>
#include <cache/cache.cuh>
#include <cache/cache_util.cuh>
#include <common/cumlHandle.hpp>
#include <common/host_buffer.hpp>
#include <cub/cub.cuh>
#include <cuda_utils.cuh>
#include <linalg/gemm.cuh>
#include <matrix/grammatrix.cuh>
#include <matrix/kernelmatrices.cuh>
#include <matrix/matrix.cuh>
#include <memory>

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
__global__ void mapColumnIndices(const int *ws, int n_ws, int n_rows,
                                 const int *unique, int n_unique, int *out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_ws) {
    int idx = ws[tid] % n_rows;
    int k = 0;
    // we have only max 1024 elements, we do a linear search
    for (int i = 0; i < n_unique; i++) {
      if (unique[i] == idx) k = i;
      // since the unique array is derived from ws, the search will always return
      // the correct idx where unique[k] == idx
    }
    out[tid] = k;
  }
}
}  // end unnamed namespace

/**
* @brief Buffer to store a kernel tile
*
* We calculate the kernel matrix for the vectors in the working set.
* For every vector x_i in the working set, we always calculate a full row of the
* kernel matrix K(x_j, x_i), j=1..n_rows.
*
* A kernel tile stores all the kernel rows for the working set, i.e. K(x_j, x_i)
* for all i in the working set, and j in 1..n_rows. For details about the kernel
* tile layout, see KernelCache::GetWSTile
*
* The kernel values can be cached to avoid repeated calculation of the kernel
* function.
*/
template <typename math_t>
class KernelCache {
 public:
  /**
   * Construct an object to manage kernel cache
   *
   * @param handle reference to cumlHandle implementation
   * @param x device array of training vectors in column major format,
   *   size [n_rows x n_cols]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param n_ws size of working set
   * @param kernel pointer to kernel (default linear)
   * @param cache_size (default 200 MiB)
   * @param svmType is this SVR or SVC
   */
  KernelCache(const cumlHandle_impl &handle, const math_t *x, int n_rows,
              int n_cols, int n_ws,
              MLCommon::Matrix::GramMatrixBase<math_t> *kernel,
              float cache_size = 200, int n_batch_in = -1,
              SvmType svmType = C_SVC, bool train = true,
              float tile_fraction = 0.5)
    : kernel(kernel),
      x(x),
      n_rows(n_rows),
      n_cols(n_cols),
      n_ws(n_ws),
      svmType(svmType),
      cublas_handle(handle.getCublasHandle()),
      d_num_selected_out(handle.getDeviceAllocator(), handle.getStream(), 1),
      d_temp_storage(handle.getDeviceAllocator(), handle.getStream()),
      // TODO try not waste memory here during prediction
      x_ws(handle.getDeviceAllocator(), handle.getStream(), n_ws * n_cols),
      x_ws2(handle.getDeviceAllocator(), handle.getStream(), n_ws * n_cols),
      tile(handle.getDeviceAllocator(), handle.getStream()),
      unique_idx(handle.getDeviceAllocator(), handle.getStream(), n_ws),
      k_col_idx(handle.getDeviceAllocator(), handle.getStream(), n_ws),
      nz_vec(handle.getDeviceAllocator(), handle.getStream(), n_ws),
      nz_idx(handle.getDeviceAllocator(), handle.getStream(), n_ws),
      permutation(handle.getDeviceAllocator(), handle.getStream(), n_ws),
      ws_cache_idx(handle.getDeviceAllocator(), handle.getStream(), n_ws),
      A_tmp(handle.getDeviceAllocator(), handle.getStream()),
      idx_tmp(handle.getDeviceAllocator(), handle.getStream()),
      train(train) {
    ASSERT(kernel != nullptr, "Kernel pointer required for KernelCache!");
    stream = handle.getStream();

    // Default kernel_column_idx map for SVC
    MLCommon::LinAlg::range(k_col_idx.data(), n_ws, stream);

    // Init cub buffers
    size_t bytes1, bytes2;
    cub::DeviceRadixSort::SortKeys(NULL, bytes1, unique_idx.data(),
                                   unique_idx.data(), n_ws, 0, sizeof(int) * 8,
                                   stream);
    cub::DeviceSelect::Unique(NULL, bytes2, unique_idx.data(),
                              unique_idx.data(), d_num_selected_out.data(),
                              n_ws, stream);
    d_temp_storage_size = max(bytes1, bytes2);
    d_temp_storage.resize(d_temp_storage_size, stream);

    n_batch = KernelCache::CalcBatchSize(n_rows, n_ws, n_batch_in,
                                         tile_fraction, &cache_size, train);
    CUML_LOG_INFO(
      "Creating KernelCache with n_rows=%d, n_ws=%d, n_batch=%d"
      ", cache_size %f MiB",
      n_rows, n_ws, n_batch, cache_size);
    tile.resize(n_ws * n_batch, stream);

    cache = std::make_unique<MLCommon::Cache::Cache<math_t>>(
      handle.getDeviceAllocator(), handle.getStream(), n_rows, cache_size);

    // Temporary buffers for KernelMV
    if (dynamic_cast<MLCommon::Matrix::RBFKernel<math_t> *>(kernel)) {
      // This is needed because ld parameter is not supported for the RBF kernel
      A_tmp.resize(n_batch * n_cols, stream);
      idx_tmp.resize(n_batch, stream);
    } else {
      // used for value/idx sorting in GetCacheIndices
      A_tmp.resize(n_ws, stream);
      idx_tmp.resize(n_ws, stream);
    }
  }

  ~KernelCache(){};

  /**
   * @brief Returns the batch size used for kernel tile evaluation
   *
   * Also returns the remaining space in buffer_size, which should be used
   * to allocate cache.
   *
   * @param n_rows number of rows
   * @param n_ws number of elements in the working set
   * @param buffer_size max size of kernel tile buffer in MiB
   * @param train wether we are in traning or prediction mode
   */
  static int CalcBatchSize(int n_rows, int n_ws, int n_batch_in,
                           float tile_fraction, float *buffer_size,
                           bool train) {
    float tile_size = train ? tile_fraction * (*buffer_size) : *buffer_size;
    tile_size *= 1024 * 1024;  // in bytes
    int n_batch =
      n_batch_in > 0 ? n_batch_in : tile_size / (n_ws * sizeof(math_t));
    n_batch = MLCommon::ceildiv(n_batch, 32);
    n_batch = min(n_rows, max(1, n_batch));  // 1 <= n_batch <=n_rows
    if (train && n_batch < n_ws) {
      // During traning we need at least n_ws * n_ws kernel tile in the inner
      // SmoBlockSolver.
      n_batch = n_ws;
    }
    CUML_LOG_INFO("Kernel tile will require %f MiB",
                  n_ws * n_batch * sizeof(math_t));
    if (n_batch_in == -1) {
      tile_size = n_batch * n_ws * sizeof(math_t) / (1024 * 1024.0);  // MiB
      *buffer_size = max(*buffer_size - tile_size, 0.0f);
    }
    if (!train) buffer_size = 0;
    return n_batch;
  }
  /**
   * @brief Get the kernel matrix tile for the working set.
   *
   * The kernel matrix is stored in column major format:
   * kernel[row_id, col_id] = kernel[row_id + col_id * n_ws]
   *
   * For SVC:
   * kernel is a square matrix with size [n_ws * n_ws], so:
   * \f[ row_id \in [0..n_ws-1], col_id \in [0..n_ws-1] \f]
   *
   * The columns correspond to the vectors in the working set. For example:
   * Let's assume thet the working set are vectors x_5, x_9, x_0, and x_4,
   * then ws_idx = [5, 9, 0 ,4]. The second column of the kernel matrix,
   * kernel[i + 1*n_ws] (\f[ i \in [0..n_ws-1] \f]), stores the kernel
   * matrix values for the second vector in the working set: K(x_i, x_9).
   *
   * For SVR:
   * We doubled the set of training vector, assigning:
   * \f[ x_{n_rows+i} = x_i for i \in [0..n_rows-1]. \f]
   *
   * The kernel matrix values are the same for x_i and x_{i+n_rows}, therefore
   * it can happen that two elements in the working set have the same x vector.
   * For example, if n_rows=10, then for ws = [5, 19, 15 0], the
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
   * The returned kernel tile array allocated/deallocated by KernelCache.
   *
   * Note: when cache_size > 0, the workspace indices can be permuted during
   * the call to GetWsTile. Use KernelCache::GetWsIndices to query the actual
   * list of workspace indices.
   *
   * @param [in] ws_idx indices of the working set
   * @return pointer to the kernel tile [ n_rows x n_unique] K(x_j, x_q)
   */
  math_t *GetWsTile(const int *ws_idx) {
    this->ws_idx = ws_idx;
    GetUniqueIndices(ws_idx, n_ws, unique_idx.data(), &n_unique);
    if (cache->GetSize() > 0) {
      int n_cached;
      cache->GetCacheIdxPartitioned(unique_idx.data(), n_unique,
                                    ws_cache_idx.data(), &n_cached, stream);
      // The kernel tile is composed of four sub tiles
      // K =  A  C
      //      B  D
      // K.shape = n_unique * n_unique
      // A.shape = n_cached * n_cached
      // B.shape = non_cached * n_cached
      // C.shape = n_cached * non_cached (C = B.T)
      // D.shape = non_cached * non_cached

      // Collect allready cached values A & B
      cache->GetVecs(ws_cache_idx.data(), n_cached, tile.data(), stream,
                     unique_idx.data(), n_unique);
      int non_cached = n_unique - n_cached;
      // We calculate tile C and D. C could be copied over from tile B, but
      // because of the column major memory layout that would lead to
      // column-wise memory copy for both C and D.
      // if (false && n_cached > 0) {
      //   MLCommon::Matrix::copyRows(x, n_rows, n_cols, x_ws.data(),
      //                              unique_idx.data(), n_unique, stream, false);
      //   MLCommon::Matrix::copyRows(x, n_rows, n_cols, x_ws2.data(),
      //                              unique_idx.data(), n_cached, stream, false);
      //   // Calculade C D
      //   (*kernel)(x_ws.data(), n_unique, n_cols, x_ws2.data(), n_cached,
      //             tile.data(), stream);
      // }
      if (non_cached > 0 && n_unique > 0) {
        int *ws_idx_new = unique_idx.data() + n_cached;
        // collect training vectors for kernel elements that needs to be calculated
        MLCommon::Matrix::copyRows(x, n_rows, n_cols, x_ws.data(),
                                   unique_idx.data(), n_unique, stream, false);
        // math_t* x2_ptr = x_ws.data() + n_cached;
        // For the above to work we would need to have ld_x2 = n_unique, but the
        // RBF kernel does not take ld parameter.
        MLCommon::Matrix::copyRows(x, n_rows, n_cols, x_ws2.data(), ws_idx_new,
                                   non_cached, stream, false);
        math_t *tile_new = tile.data() + n_cached * n_unique;
        // Calculade C D
        (*kernel)(x_ws.data(), n_unique, n_cols, x_ws2.data(), non_cached,
                  tile_new, stream);
      }
    } else {
      if (n_unique > 0) {
        // collect all the feature vectors in the working set
        MLCommon::Matrix::copyRows(x, n_rows, n_cols, x_ws.data(),
                                   unique_idx.data(), n_unique, stream, false);
        (*kernel)(x_ws.data(), n_unique, n_cols, x_ws.data(), n_unique,
                  tile.data(), stream);
      }
    }
    return tile.data();
  }

  /**
   * @brief Get the kernel tile K(A,B).
   *
   * Matrix A is a batch of rows from the feature vectors
   * A = x[row_start:row_start+n_batch, 0:ncols-1]
   *
   * If n_cached == 0, then B is simply [nB * n_cols] matrix, and we return
   * K_i,j = K(x_i, x_j) where x_i = A[i,:], x_j = B[j,:] and K is the kernel
   * function.
   *
   * If n_cached > 0 then we set
   * K_i,j = cached K_i,j if j < n_cached
   *         K(x_i, x_j) where x_i = A[i,:], x_j = B[j-n_cached,:]
   *
   * The kernel tile buffer is owned by KernelCache.
   *
   * @param row_start
   * @param n_batch batch size
   * @param matrix B, size[nB * n_cols]
   * @param nB number of rows in matrix
   *
   * @return pointer to the kernel tile [ n_batch x (n_cached + nB)]
   */
  math_t *GetTile(int row_start, int n_batch, math_t *B, int nB,
                  int n_cached = 0, int ldB = -1, int *cache_idx = nullptr) {
    if (n_cached > 0) {
      cache->GetVecs(cache_idx, n_cached, tile.data(), stream, row_start,
                     n_batch);
    }
    if (ldB == -1) ldB = nB;
    // nB corresponds to the number of non_cached elements
    if (nB > 0) {
      const math_t *A = nullptr;
      int lda = 0;
      if (dynamic_cast<MLCommon::Matrix::RBFKernel<math_t> *>(kernel)) {
        // The RBF kernel does not support ld parameters (See issue #1172)
        // To come around this limitation, we copy the batch into a temporary
        // buffer.
        thrust::counting_iterator<int> first(row_start);
        thrust::counting_iterator<int> last = first + n_batch;
        thrust::device_ptr<int> idx_ptr(idx_tmp.data());
        thrust::copy(thrust::cuda::par.on(stream), first, last, idx_ptr);
        MLCommon::Matrix::copyRows(x, n_rows, n_cols, A_tmp.data(),
                                   idx_tmp.data(), n_batch, stream, false);
        A = A_tmp.data();
        lda = n_batch;
      } else {
        A = x + row_start;
        lda = n_rows;
      }
      math_t *tile_new = tile.data() + n_cached * n_batch;
      (*kernel)(A, n_batch, n_cols, B, nB, tile_new, stream, lda, ldB, n_batch);
      if (cache_idx && cache->GetSize() > 0) {
        cache->StoreVecs(tile_new, nB, nB, cache_idx + n_cached, stream,
                         nullptr, row_start, n_batch);
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
  *   KernelCache
  */
  int *GetColIdxMap() {
    if (svmType == EPSILON_SVR) {
      mapColumnIndices<<<MLCommon::ceildiv(n_ws, TPB), TPB, 0, stream>>>(
        ws_idx, n_ws, n_rows, unique_idx.data(), n_unique, k_col_idx.data());
      CUDA_CHECK(cudaPeekAtLastError());
    }
    // else k_col_idx is already initialized properly
    return k_col_idx.data();
  }

  /**
   * @brief Return the batch size.
   */
  int GetBatchSize() { return n_batch; }

  /**
   * @brief Return the number of unique elements in the working set.
   *
   *  This is equal with the number of columns in the kernel tile.
   */
  int GetUniqueSize() { return n_unique; }

  const int *GetWsIndices() {
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
  void GetVecIndices(const int *ws_idx, int n_ws, int *vec_idx) {
    int n = n_rows;
    MLCommon::LinAlg::unaryOp(
      vec_idx, ws_idx, n_ws,
      [n] __device__(math_t y) { return y < n ? y : y - n; }, stream);
  }

  /**
   * Return the number of unique indices in the working set.
   *
   * For SVC n_unique == n_ws, for SVR n_unique <= n_ws.
   */
  int *GetUniqueIndices(int *ws_idx = nullptr, int n_ws = 0) {
    if (ws_idx) {
      GetUniqueIndices(ws_idx, n_ws, unique_idx.data(), &n_unique);
    }
    return unique_idx.data();
  }

  /**
   * Look up indices in the cache.
   *
   * The cache routines permute the indices:
   * - partition cached and non_cached indices
   * - sort non cached indices by cache set
   *
   * We consider idx, val as pairs and permute the two arrays the same way.
   *
   * @param idx indices, subset of the working set size[n], 0 <= idx[i] < n_rows
   * @param val values, size [n]
   * @param cache_idx the cache indices corresponding to, size [n]. On exit
   *   for i < n_cached cache_idx[i] stores the actual cache index for idx[i]
   *   for i >= n_cached cache_idx[i] stores the newly assigned cache index
   *
   * @return number of cached elements
   */
  int GetCacheIndices(int *idx, int n, math_t *val, int *cache_idx) {
    if (n > 0 && cache->GetSize() > 0) {
      int n_cached;
      // We create a copy of the original index order, so that we can reorder
      // val accordingly.

      MLCommon::copy(idx_tmp.data(), idx, n, stream);
      MLCommon::copy(A_tmp.data(), val, n, stream);

      cache->GetCacheIdxPartitioned(idx, n, cache_idx, &n_cached, stream);

      int non_cached = n - n_cached;
      if (non_cached > 0) {
        cache->AssignCacheIdx(idx + n_cached, non_cached, cache_idx + n_cached,
                              stream);
      }
      // Find a permutation from idx_tmp -> idx
      mapColumnIndices<<<MLCommon::ceildiv(n, TPB), TPB, 0, stream>>>(
        idx, n, n_rows, idx_tmp.data(), n, permutation.data());
      CUDA_CHECK(cudaPeekAtLastError());
      // permute val according to permutation
      thrust::device_ptr<math_t> val_ptr(A_tmp.data());
      thrust::device_ptr<math_t> val_ptr_permuted(val);
      thrust::device_ptr<int> perm_ptr(permutation.data());
      thrust::copy(thrust::cuda::par.on(stream),
                   thrust::make_permutation_iterator(val_ptr, perm_ptr),
                   thrust::make_permutation_iterator(val_ptr, perm_ptr + n),
                   val_ptr_permuted);
      return n_cached;
    }
    return 0;
  }

  /** @brief Helper function to prepare a feature matrix from the working set.
   *
   * The matrix contains the training vectors that correspond to nonzero
   * elements af dalpha, omitting those rows that are superfluous due to
   * caching.
   *
   * The returned B matrix will be used to construct the kernel matrix
   * K_i,j = A_i,l * B_j,l where i = 0..n_rows-1, j=0..n_B
   *
   * Without cache and if every elements in delta_alpha is nonzero, then it
   * returns all the training vectors in the working set as an [n_ws*n_cols]
   * matrix (in column major layout):
   * B[i + k*n_ws] = x[unique_idx[i] + k*n_rows] for k=0..n_cols-1,
   *                                                 i=0..n_unique-1
   *
   * Still without cache, but assuming that dalpha has zero elements: we omit
   * rows of B that corresponds to zeros in dalpha. In numpy notation
   * B_nz = B[dalpha!=0, :]. The indices of the nonzero elements of dalpha are
   * stored in nz_idx member variable, the number of its elements is n_nz.
   *
   * We have a cache to store column vectors of K. We do not need to calculate
   * K[:,j] if it is stored in the cache, and it implies that B[j,:] values
   * will not be used if column j is cached. We omit such rows and return the
   * following matrix.
   * B_final = B_nz[!cached(nz_idx), :]
   *
   * On exit:
   *  - x_ws is populated by B_final,
   *  - nz_vec stores the nonzero elements of dalpha
   *  - nz_idx[0:n_cached] indices of ws vectors where dalpha is nonzero, and
   *    where the corresponding Kernel matrix elements are cached
   *  - ws_cache_idx stores the corresponding cache indices
   *  - nz_idx[n_cached:] indicices of ws vectors that stored in B_final
   *
   * n_cached = nnz - nB is not returned
   *
   * @param dalpha input vector size [n_ws]
   * @param B output matrix constructed from the training vectors in the working
   *    set [n_ws * nB]
   * @param nB number of elements
   * @param GetCacheIndices
   * @return pointer to B_final
   */
  math_t *GetWsVecForDeltaAlpha(const math_t *dalpha, int *n_nz, int *nB,
                                int *ldB) {
    // We consider only the unique elements of the working set. Make sure
    // unique_idx is defined before calling KernelMV (during training it is
    // defined by calling GetWsTile).
    GetNonzeroDeltaAlpha(dalpha, GetUniqueSize(), GetUniqueIndices(),
                         nz_vec.data(), n_nz, nz_idx.data());
    int n_cached =
      GetCacheIndices(nz_idx.data(), *n_nz, nz_vec.data(), ws_cache_idx.data());
    // collect all the feature vectors for the nonzero & non-cached elements
    // in the working
    *nB = *n_nz - n_cached;
    *ldB = *nB;  // it should be n_ws if all kernels would support ld param
    int *indices = nz_idx.data() + n_cached;
    MLCommon::Matrix::copyRows(x, n_rows, n_cols, x_ws.data(), indices, *nB,
                               stream, false, *ldB);
    return x_ws.data();
  }

  /** Matrix vector multiplication where the matrix is the Kernel matrix.
   *
   * f = K(x,D)*v + beta * f
   *
   * where x is the set of all train / predic vectors (given to the consturctor
   * of kernel cache as parameter x, size [n_rows, n_cols]).
   *
   * D = [C; B] where C is the feature matix of those traning vectors that
   * correspond to chaced columns of K.  B is another set of feature vectors
   *
   * @param v input vector, size [n]
   * @param B feature vectors B size [n_B*n_cols]
   * @param stride used for B
   * @param beta scalar factor
   * @param f input/output values, size [n]
   * @param train whether we are in training or prediction mode
   */
  void KernelMV(const math_t *v, int n, math_t *B, int nB, int ldB, math_t beta,
                math_t *f) {
    int n_cached = n - nB;
    ASSERT(n_cached >= 0, "Error in KernelMV, inconsistent array sizes");
    if (n == 0) {
      // No need for GEMV
      if (beta != 0) {
        MLCommon::LinAlg::unaryOp(
          f, f, n, [beta] __device__(math_t y) { return beta * y; }, stream);
      }
      return;
    }
    // We process the input data batchwise:
    //  - calculate the kernel values K[x_batch, B]
    //  - calculate f(x_batch) = K[x_batch, x_idx] * v + beta*f
    int n_batch = this->n_batch;
    for (int i = 0; i < n_rows; i += n_batch) {
      if (i + n_batch >= n_rows) {
        n_batch = n_rows - i;
      }
      math_t *K =
        GetTile(i, n_batch, B, nB, n_cached, ldB, ws_cache_idx.data());
      math_t one = 1;
      CUBLAS_CHECK(MLCommon::LinAlg::cublasgemv(cublas_handle, CUBLAS_OP_N,
                                                n_batch, n, &one, K, n_batch, v,
                                                1, &beta, f + i, 1, stream));
      if (train && svmType == EPSILON_SVR) {
        // SVR has doubled the number of trainig vectors and we need to update
        // alpha for both batches individually
        CUBLAS_CHECK(MLCommon::LinAlg::cublasgemv(
          cublas_handle, CUBLAS_OP_N, n_batch, n, &one, K, n_batch, v, 1, &beta,
          f + i + n_rows, 1, stream));
      }
    }
  }

  /** Matrix vector multiplication where the matrix is the Kernel matrix.
  *
  *  Same as above, but matrix B is derived from the working set.
  */
  void KernelMV(const math_t *v, int n, math_t beta, math_t *f) {
    int n_nz;
    int nB;
    int ldB;
    math_t *B = GetWsVecForDeltaAlpha(v, &n_nz, &nB, &ldB);
    KernelMV(nz_vec.data(), n_nz, B, nB, ldB, beta, f);
  }

  /** @brief Select nonzero coefficients and their indices.
   *
   * The kernel function calculation is time consuming therefore we only
   * want to calulate that for nonzero vector elements. This function
   * select the nonzeros from vec, and returns the corresponding workspace
   * indices plus
   *
   * @param vec on size [n_ws]
   * @param n_ws number of workspace elements
   * @param idx workspace indices, size[n_ws], 0 <= idx[i] <= n_rows-1
   * @param nz_vec nonzero values from vec, size [n_nz]
   * @param n_nz number of nonzero elements in vec
   * @param nz_idx workspace indices of the nonzero elements size [n_nz]
   */
  void GetNonzeroDeltaAlpha(const math_t *vec, int n_ws, const int *idx,
                            math_t *nz_vec, int *n_nz, int *nz_idx) {
    thrust::device_ptr<math_t> vec_ptr(const_cast<math_t *>(vec));
    thrust::device_ptr<math_t> nz_vec_ptr(nz_vec);
    thrust::device_ptr<int> idx_ptr(const_cast<int *>(idx));
    thrust::device_ptr<int> nz_idx_ptr(nz_idx);
    auto nonzero = [] __device__(math_t a) { return a != 0; };
    thrust::device_ptr<int> nz_end =
      thrust::copy_if(thrust::cuda::par.on(stream), idx_ptr, idx_ptr + n_ws,
                      vec_ptr, nz_idx_ptr, nonzero);
    *n_nz = nz_end - nz_idx_ptr;
    thrust::copy_if(thrust::cuda::par.on(stream), vec_ptr, vec_ptr + n_ws,
                    nz_vec_ptr, nonzero);
  }

 public:              //private:
  const math_t *x;    //!< pointer to the training vectors
  const int *ws_idx;  //!< pointer to the working set indices

  /// feature vectors in the current working set
  MLCommon::device_buffer<math_t> x_ws;
  MLCommon::device_buffer<math_t> x_ws2;
  /// cache position of a workspace vectors
  MLCommon::device_buffer<int> ws_cache_idx;

  MLCommon::device_buffer<math_t> tile;  //!< Kernel matrix  tile

  int n_rows;    //!< number of rows in x
  int n_cols;    //!< number of columns in x
  int n_ws;      //!< number of elements in the working set
  int n_unique;  //!< number of unique x vectors in the working set
  int n_batch;   //!< batch size for batched kernel matrix evaluation

  bool train;  //!< whether we are in training mode or prodiction mode

  cublasHandle_t cublas_handle;

  MLCommon::Matrix::GramMatrixBase<math_t> *kernel;

  const cumlHandle_impl handle;

  const int TPB = 256;  //!< threads per block for kernels launched

  std::unique_ptr<MLCommon::Cache::Cache<math_t>> cache;

  cudaStream_t stream;
  SvmType svmType;
  MLCommon::device_buffer<int> unique_idx;  //!< Training vector indices
  /// Column index map for the kernel tile
  MLCommon::device_buffer<int> k_col_idx;

  /// Helper arrays for KernelMV
  MLCommon::device_buffer<math_t> A_tmp;
  MLCommon::device_buffer<int> idx_tmp;
  MLCommon::device_buffer<math_t> nz_vec;  //!< nonzero values in KernelMV
  MLCommon::device_buffer<int> nz_idx;     //!< indices of nonzeros in KernelMV
  MLCommon::device_buffer<int>
    permutation;  //!< temporary array for cache lookup

  // Helper arrays for cub
  MLCommon::device_buffer<int> d_num_selected_out;
  MLCommon::device_buffer<char> d_temp_storage;
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
  void GetUniqueIndices(const int *ws_idx, int n_ws, int *unique_idx,
                        int *n_unique) {
    if (svmType == C_SVC) {
      *n_unique = n_ws;
      MLCommon::copy(unique_idx, ws_idx, n_ws, stream);
      return;
    }
    // for EPSILON_SVR
    GetVecIndices(ws_idx, n_ws, unique_idx);
    cub::DeviceRadixSort::SortKeys(d_temp_storage.data(), d_temp_storage_size,
                                   unique_idx, ws_cache_idx.data(), n_ws, 0,
                                   sizeof(int) * 8, stream);
    cub::DeviceSelect::Unique(d_temp_storage.data(), d_temp_storage_size,
                              ws_cache_idx.data(), unique_idx,
                              d_num_selected_out.data(), n_ws, stream);
    MLCommon::updateHost(n_unique, d_num_selected_out.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

};  // end namespace SVM
};  // end namespace ML
