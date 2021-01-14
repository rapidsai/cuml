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

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <raft/device_atomics.cuh>

#include <iostream>
#define restrict __restrict__

#pragma once

namespace raft {
namespace sparse {

/** @brief A Container object for sparse coordinate. There are two motivations
 * behind using a container for COO arrays.
 *
 * The first motivation is that it simplifies code, rather than always having
 * to pass three arrays as function arguments.
 *
 * The second is more subtle, but much more important. The size
 * of the resulting COO from a sparse operation is often not known ahead of time,
 * since it depends on the contents of the underlying graph. The COO object can
 * allocate the underlying arrays lazily so that the object can be created by the
 * user and passed as an output argument in a sparse primitive. The sparse primitive
 * would have the responsibility for allocating and populating the output arrays,
 * while the original caller still maintains ownership of the underlying memory.
 *
 * @tparam T: the type of the value array.
 * @tparam Index_Type: the type of index array
 *
 */
template <typename T, typename Index_Type = int>
class COO {
 protected:
  raft::mr::device::buffer<Index_Type> rows_arr;
  raft::mr::device::buffer<Index_Type> cols_arr;
  raft::mr::device::buffer<T> vals_arr;

 public:
  Index_Type nnz;
  Index_Type n_rows;
  Index_Type n_cols;

  /**
    * @param d_alloc: the device allocator to use for the underlying buffers
    * @param stream: CUDA stream to use
    */
  COO(std::shared_ptr<raft::mr::device::allocator> d_alloc, cudaStream_t stream)
    : rows_arr(d_alloc, stream, 0),
      cols_arr(d_alloc, stream, 0),
      vals_arr(d_alloc, stream, 0),
      nnz(0),
      n_rows(0),
      n_cols(0) {}

  /**
    * @param rows: coo rows array
    * @param cols: coo cols array
    * @param vals: coo vals array
    * @param nnz: size of the rows/cols/vals arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of cols in the dense matrix
    */
  COO(raft::mr::device::buffer<Index_Type> &rows,
      raft::mr::device::buffer<Index_Type> &cols,
      raft::mr::device::buffer<T> &vals, Index_Type nnz, Index_Type n_rows = 0,
      Index_Type n_cols = 0)
    : rows_arr(rows),
      cols_arr(cols),
      vals_arr(vals),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols) {}

  /**
    * @param d_alloc: the device allocator use
    * @param stream: CUDA stream to use
    * @param nnz: size of the rows/cols/vals arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of cols in the dense matrix
    * @param init: initialize arrays with zeros
    */
  COO(std::shared_ptr<raft::mr::device::allocator> d_alloc, cudaStream_t stream,
      Index_Type nnz, Index_Type n_rows = 0, Index_Type n_cols = 0,
      bool init = true)
    : rows_arr(d_alloc, stream, nnz),
      cols_arr(d_alloc, stream, nnz),
      vals_arr(d_alloc, stream, nnz),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols) {
    if (init) init_arrays(stream);
  }

  void init_arrays(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(this->rows_arr.data(), 0,
                               this->nnz * sizeof(Index_Type), stream));
    CUDA_CHECK(cudaMemsetAsync(this->cols_arr.data(), 0,
                               this->nnz * sizeof(Index_Type), stream));
    CUDA_CHECK(
      cudaMemsetAsync(this->vals_arr.data(), 0, this->nnz * sizeof(T), stream));
  }

  ~COO() {}

  /**
    * @brief Size should be > 0, with the number of rows
    * and cols in the dense matrix being > 0.
    */
  bool validate_size() const {
    if (this->nnz < 0 || n_rows < 0 || n_cols < 0) return false;
    return true;
  }

  /**
    * @brief If the underlying arrays have not been set,
    * return false. Otherwise true.
    */
  bool validate_mem() const {
    if (this->rows_arr.size() == 0 || this->cols_arr.size() == 0 ||
        this->vals_arr.size() == 0) {
      return false;
    }

    return true;
  }

  /*
   * @brief Returns the rows array
   */
  Index_Type *rows() { return this->rows_arr.data(); }

  /**
   * @brief Returns the cols array
   */
  Index_Type *cols() { return this->cols_arr.data(); }

  /**
   * @brief Returns the vals array
   */
  T *vals() { return this->vals_arr.data(); }

  /**
    * @brief Send human-readable state information to output stream
    */
  friend std::ostream &operator<<(std::ostream &out,
                                  const COO<T, Index_Type> &c) {
    if (c.validate_size() && c.validate_mem()) {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

      out << raft::arr2Str(c.rows_arr.data(), c.nnz, "rows", stream)
          << std::endl;
      out << raft::arr2Str(c.cols_arr.data(), c.nnz, "cols", stream)
          << std::endl;
      out << raft::arr2Str(c.vals_arr.data(), c.nnz, "vals", stream)
          << std::endl;
      out << "nnz=" << c.nnz << std::endl;
      out << "n_rows=" << c.n_rows << std::endl;
      out << "n_cols=" << c.n_cols << std::endl;

      CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
      out << "Cannot print COO object: Uninitialized or invalid." << std::endl;
    }

    return out;
  }

  /**
    * @brief Set the number of rows and cols
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of columns in the dense matrix
    */
  void setSize(int n_rows, int n_cols) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
  }

  /**
    * @brief Set the number of rows and cols for a square dense matrix
    * @param n: number of rows and cols
    */
  void setSize(int n) {
    this->n_rows = n;
    this->n_cols = n;
  }

  /**
    * @brief Allocate the underlying arrays
    * @param nnz: size of underlying row/col/val arrays
    * @param init: should values be initialized to 0?
    * @param stream: CUDA stream to use
    */
  void allocate(int nnz, bool init, cudaStream_t stream) {
    this->allocate(nnz, 0, init, stream);
  }

  /**
    * @brief Allocate the underlying arrays
    * @param nnz: size of the underlying row/col/val arrays
    * @param size: the number of rows/cols in a square dense matrix
    * @param init: should values be initialized to 0?
    * @param stream: CUDA stream to use
    */
  void allocate(int nnz, int size, bool init, cudaStream_t stream) {
    this->allocate(nnz, size, size, init, stream);
  }

  /**
    * @brief Allocate the underlying arrays
    * @param nnz: size of the underlying row/col/val arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of columns in the dense matrix
    * @param init: should values be initialized to 0?
    * @param stream: stream to use for init
    */
  void allocate(int nnz, int n_rows, int n_cols, bool init,
                cudaStream_t stream) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->nnz = nnz;

    this->rows_arr.resize(this->nnz, stream);
    this->cols_arr.resize(this->nnz, stream);
    this->vals_arr.resize(this->nnz, stream);

    if (init) init_arrays(stream);
  }
};

};  // namespace sparse
};  // namespace raft
