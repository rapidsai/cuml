/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

//#include <raft/util/cuda_utils.cuh>
//#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Matrix {

template <typename math_t>
class DenseMatrix;
template <typename math_t>
class CsrMatrix;
template <typename math_t>
class CooMatrix;
template <typename math_t>
class ResizableCsrMatrix;

enum MatrixType { DENSE, CSR, COO };

template <typename math_t>
class Matrix {
 public:
  Matrix(int rows, int cols) : n_rows(rows), n_cols(cols){};
  int numRows() const { return n_rows; };
  int numCols() const { return n_cols; };
  virtual MatrixType getType() const = 0;
  virtual ~Matrix(){};

  DenseMatrix<math_t>* asDense()
  {
    DenseMatrix<math_t>* cast = dynamic_cast<DenseMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  CsrMatrix<math_t>* asCsr()
  {
    CsrMatrix<math_t>* cast = dynamic_cast<CsrMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  ResizableCsrMatrix<math_t>* asResizableCsr()
  {
    ResizableCsrMatrix<math_t>* cast = dynamic_cast<ResizableCsrMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  CooMatrix<math_t>* asCoo()
  {
    CooMatrix<math_t>* cast = dynamic_cast<CooMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  const DenseMatrix<math_t>* asDense() const
  {
    const DenseMatrix<math_t>* cast = dynamic_cast<const DenseMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  const CsrMatrix<math_t>* asCsr() const
  {
    const CsrMatrix<math_t>* cast = dynamic_cast<const CsrMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  const ResizableCsrMatrix<math_t>* asResizableCsr() const
  {
    const ResizableCsrMatrix<math_t>* cast = dynamic_cast<const ResizableCsrMatrix<math_t>*>(this);
    return cast;
  };

  const CooMatrix<math_t>* asCoo() const
  {
    const CooMatrix<math_t>* cast = dynamic_cast<const CooMatrix<math_t>*>(this);
    assert(cast != nullptr);
    return cast;
  };

  int n_rows;
  int n_cols;
};

template <typename math_t>
class DenseMatrix : public Matrix<math_t> {
 public:
  DenseMatrix(math_t* data, int rows, int cols) : Matrix<math_t>(rows, cols), data(data) {}
  virtual MatrixType getType() const { return MatrixType::DENSE; }
  math_t* data;
};

template <typename math_t>
class CsrMatrix : public Matrix<math_t> {
 public:
  CsrMatrix(int* indptr,
            int* indices,
            math_t* data,
            int nnz,
            int rows,
            int cols,
            int row_capacity = -1,
            int nnz_capacity = -1)
    : Matrix<math_t>(rows, cols),
      indptr(indptr),
      indices(indices),
      data(data),
      nnz(nnz),
      row_capacity(row_capacity),
      nnz_capacity(nnz_capacity)
  {
    if (row_capacity == -1) row_capacity = rows;
    if (nnz_capacity == -1) nnz_capacity = nnz;
  }
  virtual MatrixType getType() const { return MatrixType::CSR; }

  int nnz;
  int* indptr;
  int* indices;
  math_t* data;

  int nnz_capacity;
  int row_capacity;
};

template <typename math_t>
class ResizableCsrMatrix : public CsrMatrix<math_t> {
 public:
  ResizableCsrMatrix(int rows, int cols, int nnz, cudaStream_t stream)
    : CsrMatrix<math_t>(nullptr, nullptr, nullptr, nnz, rows, cols),
      stream(stream),
      d_indptr(rows + 1, stream),
      d_indices(nnz, stream),
      d_data(nnz, stream)
  {
    CsrMatrix<math_t>::indptr  = d_indptr.data();
    CsrMatrix<math_t>::indices = d_indices.data();
    CsrMatrix<math_t>::data    = d_data.data();
  }

  void reserveRows(int rows)
  {
    d_indptr.reserve(rows + 1, stream);
    if (rows > CsrMatrix<math_t>::row_capacity) CsrMatrix<math_t>::row_capacity = rows;
    CsrMatrix<math_t>::indptr = d_indptr.data();
  }

  void reserveNnz(int nnz)
  {
    d_indices.reserve(nnz, stream);
    d_data.reserve(nnz, stream);
    if (nnz > CsrMatrix<math_t>::nnz_capacity) CsrMatrix<math_t>::nnz_capacity = nnz;
    CsrMatrix<math_t>::indices = d_indices.data();
    CsrMatrix<math_t>::data    = d_data.data();
  }

  cudaStream_t stream;
  rmm::device_uvector<int> d_indptr, d_indices;
  rmm::device_uvector<math_t> d_data;
};

template <typename math_t>
class CooMatrix : public Matrix<math_t> {
 public:
  CooMatrix(int* rowindex, int* colindex, math_t* data, int nnz, int rows, int cols)
    : Matrix<math_t>(rows, cols), rowindex(rowindex), colindex(colindex), data(data)
  {
  }
  virtual MatrixType getType() const { return MatrixType::COO; }
  int* rowindex;
  int* colindex;
  math_t* data;
};

};  // end namespace Matrix
};  // end namespace MLCommon
