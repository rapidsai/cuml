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
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Matrix {

template <typename math_t>
class Matrix;

template <typename math_t>
class DenseMatrix : public Matrix<math_t> {
 public:
  using dense_matrix_view_t = raft::device_matrix_view<math_t, int, raft::layout_stride>;
  using dense_matrix_const_view_t =
    raft::device_matrix_view<const math_t, int, raft::layout_stride>;

  // create a non-owning wrapper
  DenseMatrix(math_t* data, int rows, int cols, bool row_major = false, int ld = 0)
    : Matrix<math_t>()
  {
    update_dense_matrix_view(data, rows, cols, row_major, ld);
  }

  // create a data-owning wrapper
  DenseMatrix(const raft::handle_t& handle, int rows, int cols, bool row_major = false)
    : Matrix<math_t>(),
      d_data_(std::make_unique<rmm::device_uvector<math_t>>(rows * cols, handle.get_stream()))
  {
    update_dense_matrix_view(d_data_->data(), rows, cols, row_major);
  }

  bool is_dense() const { return true; }
  int get_n_rows() const { return dense_matrix_view_->extent(0); }
  int get_n_cols() const { return dense_matrix_view_->extent(1); }

  bool is_row_major() const { return dense_matrix_view_->stride(0) > 1; }
  int get_ld() const
  {
    return is_row_major() ? dense_matrix_view_->stride(0) : dense_matrix_view_->stride(1);
  }

  math_t* get_data() const { return get_dense_view().data_handle(); }

  void initialize_dimensions(const raft::handle_t& handle, int rows, int cols)
  {
    assert(d_data_);
    d_data_->resize(rows * cols, handle.get_stream());
    update_dense_matrix_view(d_data_->data(), rows, cols, is_row_major());
  }

  dense_matrix_view_t get_dense_view() const { return *dense_matrix_view_; }

  dense_matrix_const_view_t get_const_dense_view() const
  {
    return raft::make_const_mdspan(*dense_matrix_view_);
  }

 private:
  void update_dense_matrix_view(
    math_t* data, int rows, int cols, bool row_major = false, int ld = 0)
  {
    if (row_major) {
      dense_matrix_view_ = std::make_unique<dense_matrix_view_t>(
        raft::make_device_strided_matrix_view<math_t, int, raft::layout_c_contiguous>(
          data, rows, cols, ld));
    } else {
      dense_matrix_view_ = std::make_unique<dense_matrix_view_t>(
        raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
          data, rows, cols, ld));
    }
  }

 private:
  std::unique_ptr<dense_matrix_view_t> dense_matrix_view_;
  std::unique_ptr<rmm::device_uvector<math_t>> d_data_;
};

template <typename math_t>
class CsrMatrix : public Matrix<math_t> {
 public:
  using csr_matrix_view_t       = raft::device_csr_matrix_view<math_t, int, int, int>;
  using csr_matrix_const_view_t = raft::device_csr_matrix_view<const math_t, int, int, int>;

  // create a non-owning wrapper
  CsrMatrix(int* indptr, int* indices, math_t* data, int nnz, int rows, int cols) : Matrix<math_t>()
  {
    auto csr_structure =
      raft::make_device_csr_structure_view<int, int, int>(indptr, indices, rows, cols, nnz);
    csr_matrix_view_ = std::make_unique<csr_matrix_view_t>(
      raft::device_span<math_t>(data, csr_structure.get_nnz()), csr_structure);
  }

  // create a data-owning wrapper
  CsrMatrix(const raft::handle_t& handle, int nnz, int rows, int cols)
    : Matrix<math_t>(),
      d_indptr_(std::make_unique<rmm::device_uvector<int>>(rows + 1, handle.get_stream())),
      d_indices_(std::make_unique<rmm::device_uvector<int>>(nnz, handle.get_stream())),
      d_data_(std::make_unique<rmm::device_uvector<math_t>>(nnz, handle.get_stream()))
  {
    auto csr_structure = raft::make_device_csr_structure_view<int, int, int>(
      d_indptr_->data(), d_indices_->data(), rows, cols, nnz);
    csr_matrix_view_ = std::make_unique<csr_matrix_view_t>(
      raft::device_span<math_t>(d_data_->data(), csr_structure.get_nnz()), csr_structure);
  }

  bool is_dense() const { return false; }
  int get_n_rows() const { return get_csr_view().get_structure().get_n_rows(); }
  int get_n_cols() const { return get_csr_view().get_structure().get_n_cols(); }
  int get_nnz() const { return get_csr_view().get_structure().get_nnz(); }

  int* get_indptr() const { return get_csr_view().get_structure().get_indptr().data(); }
  int* get_indices() const { return get_csr_view().get_structure().get_indices().data(); }
  math_t* get_data() const { return get_csr_view().get_elements().data(); }

  void initialize_dimensions(const raft::handle_t& handle, int rows, int cols)
  {
    assert(d_indptr_);
    d_indptr_->resize(rows + 1, handle.get_stream());
    auto csr_structure = raft::make_device_csr_structure_view<int, int, int>(
      d_indptr_->data(), d_indices_->data(), rows, cols, get_nnz());
    csr_matrix_view_ = std::make_unique<csr_matrix_view_t>(
      raft::device_span<math_t>(d_data_->data(), csr_structure.get_nnz()), csr_structure);
  }

  void initialize_sparsity(const raft::handle_t& handle, int nnz)
  {
    assert(d_indices_);
    assert(d_data_);
    d_indices_->resize(nnz, handle.get_stream());
    d_data_->resize(nnz, handle.get_stream());
    auto csr_structure = raft::make_device_csr_structure_view<int, int, int>(
      d_indptr_->data(), d_indices_->data(), get_n_rows(), get_n_cols(), nnz);
    csr_matrix_view_ = std::make_unique<csr_matrix_view_t>(
      raft::device_span<math_t>(d_data_->data(), csr_structure.get_nnz()), csr_structure);
  }

  csr_matrix_view_t get_csr_view() const { return *csr_matrix_view_; }

  csr_matrix_const_view_t get_const_csr_view() const
  {
    csr_matrix_const_view_t const_view(raft::device_span<const math_t>(get_data(), get_nnz()),
                                       get_csr_view().get_structure());
    return const_view;
  }

  std::unique_ptr<csr_matrix_view_t> csr_matrix_view_;

  // unfortunately cannot utilize data owning raft::csr_matrix yet as it does not support change in
  // dimensions rows/cols
  std::unique_ptr<rmm::device_uvector<int>> d_indptr_, d_indices_;
  std::unique_ptr<rmm::device_uvector<math_t>> d_data_;
};

/*
 * Thin matrix wrapper to allow single API for different matrix representations
 */
template <typename math_t>
class Matrix {
 public:
  // Matrix(int rows, int cols) : n_rows(rows), n_cols(cols){};
  virtual ~Matrix(){};

  virtual bool is_dense() const  = 0;
  virtual int get_n_rows() const = 0;
  virtual int get_n_cols() const = 0;

  virtual void initialize_dimensions(const raft::handle_t& handle, int rows, int cols) = 0;

  DenseMatrix<math_t>* as_dense()
  {
    DenseMatrix<math_t>* cast = dynamic_cast<DenseMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  CsrMatrix<math_t>* as_csr()
  {
    CsrMatrix<math_t>* cast = dynamic_cast<CsrMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  const DenseMatrix<math_t>* as_dense() const
  {
    const DenseMatrix<math_t>* cast = dynamic_cast<const DenseMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  const CsrMatrix<math_t>* as_csr() const
  {
    const CsrMatrix<math_t>* cast = dynamic_cast<const CsrMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };
};

template <typename math_t>
class DenseMatrix3;
template <typename math_t>
class CsrMatrix3;

/*
 * Thin matrix wrapper to allow single API for different matrix representations
 */
template <typename math_t>
class Matrix3 {
 public:
  Matrix3(int rows, int cols) : n_rows(rows), n_cols(cols){};
  virtual bool isDense() const = 0;
  virtual ~Matrix3(){};

  DenseMatrix3<math_t>* asDense()
  {
    DenseMatrix3<math_t>* cast = dynamic_cast<DenseMatrix3<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  CsrMatrix3<math_t>* asCsr()
  {
    CsrMatrix3<math_t>* cast = dynamic_cast<CsrMatrix3<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  const DenseMatrix3<math_t>* asDense() const
  {
    const DenseMatrix3<math_t>* cast = dynamic_cast<const DenseMatrix3<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  const CsrMatrix3<math_t>* asCsr() const
  {
    const CsrMatrix3<math_t>* cast = dynamic_cast<const CsrMatrix3<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  int n_rows;
  int n_cols;
};

template <typename math_t>
class DenseMatrix3 : public Matrix3<math_t> {
 public:
  DenseMatrix3(math_t* data, int rows, int cols, bool row_major = false, int ld_in = 0)
    : Matrix3<math_t>(rows, cols), data(data), is_row_major(row_major), ld(ld_in)
  {
    if (ld <= 0) ld = is_row_major ? cols : rows;
  }
  bool isDense() const { return true; }
  math_t* data;
  bool is_row_major;
  int ld;
};

template <typename math_t>
class CsrMatrix3 : public Matrix3<math_t> {
 public:
  CsrMatrix3(int* indptr, int* indices, math_t* data, int nnz, int rows, int cols)
    : Matrix3<math_t>(rows, cols), indptr(indptr), indices(indices), data(data), nnz(nnz)
  {
  }
  bool isDense() const { return false; }

  int nnz;
  int* indptr;
  int* indices;
  math_t* data;
};

/*
 * Extension to raft matrix wrapper that owns the backing memory
 * and allows dynamic resizing. This simplifies CSR row extraction
 * where the target nnz is not known in advance
 */
template <typename math_t>
class ResizableCsrMatrix3 : public CsrMatrix3<math_t> {
 public:
  ResizableCsrMatrix3(int rows, int cols, int nnz, cudaStream_t stream)
    : CsrMatrix3<math_t>(nullptr, nullptr, nullptr, nnz, rows, cols),
      d_indptr(rows + 1, stream),
      d_indices(nnz, stream),
      d_data(nnz, stream)
  {
    CsrMatrix3<math_t>::indptr  = d_indptr.data();
    CsrMatrix3<math_t>::indices = d_indices.data();
    CsrMatrix3<math_t>::data    = d_data.data();
  }

  void reserveRows(int rows, cudaStream_t stream)
  {
    d_indptr.reserve(rows + 1, stream);
    CsrMatrix3<math_t>::indptr = d_indptr.data();
  }

  void reserveNnz(int nnz, cudaStream_t stream)
  {
    d_indices.reserve(nnz, stream);
    d_data.reserve(nnz, stream);
    CsrMatrix3<math_t>::indices = d_indices.data();
    CsrMatrix3<math_t>::data    = d_data.data();
  }

  rmm::device_uvector<int> d_indptr, d_indices;
  rmm::device_uvector<math_t> d_data;
};

};  // end namespace Matrix
};  // end namespace MLCommon