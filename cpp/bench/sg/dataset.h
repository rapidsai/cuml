/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <linalg/cublas_wrappers.h>
#include <random/make_blobs.h>
#include <common/cumlHandle.hpp>
#include <cuML.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ML {
namespace Bench {

/**
 * Indicates the dataset size. This is supposed to be used as the base class
 * by every Benchmark's Params structure.
 */
struct DatasetParams {
  /** number of rows in the datset */
  int nrows;
  /** number of cols in the dataset */
  int ncols;
  /** number of classes in the dataset (useless for regression cases) */
  int nclasses;
  /** input dataset is stored row or col major? */
  bool rowMajor;

  std::string str() const {
    std::ostringstream oss;
    oss << "nrows=" << nrows << ";ncols=" << ncols << ";nclasses=" << nclasses
        << ";rowMajor=" << rowMajor;
    return oss.str();
  }
};

/** Holds params needed to generate blobs dataset */
template <typename D>
struct BlobsParams : public DatasetParams {
  D cluster_std;
  bool shuffle;
  D center_box_min, center_box_max;
  uint64_t seed;

  std::string str() const {
    std::ostringstream oss;
    oss << ";cluster_std=" << cluster_std << ";shuffle=" << shuffle
        << ";center_box_min=" << center_box_min
        << ";center_box_max=" << center_box_max << ";seed=" << seed;
    return DatasetParams::str() + oss.str();
  }
};

/**
 * @brief A simple object to hold the loaded dataset for benchmarking
 * @tparam D type of the dataset (type of X)
 * @tparam L type of the labels/output (type of y)
 */
template <typename D, typename L>
struct Dataset : public DatasetParams {
  /** input data */
  D* X;
  /** labels or output associated with each row of input data */
  L* y;

  /** free-up the buffers */
  void deallocate(const cumlHandle& handle) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    allocator->deallocate(X, nrows * ncols * sizeof(D), stream);
    allocator->deallocate(y, nrows * sizeof(L), stream);
  }

  /** whether the current dataset is for classification or regression */
  bool isClassification() const { return typeid(D) != typeid(L); }

  /** generate random blobs data. Args meaning is the same as in make_blobs */
  void blobs(const cumlHandle& handle, int rows, int cols, bool isRowMajor,
             int nclass, D cluster_std, bool shuffle, D center_box_min,
             D center_box_max, uint64_t seed) {
    ASSERT(isClassification(),
           "make_blobs: is only for classification/clustering problems!");
    allocate(handle, rows, cols);
    rowMajor = isRowMajor;
    nclasses = nclass;
    D* tmpX;
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    if (!isRowMajor) {
      tmpX = (D*)allocator->allocate(nrows * ncols * sizeof(D), stream);
    } else {
      tmpX = X;
    }
    MLCommon::Random::make_blobs<D, int>(
      tmpX, y, nrows, ncols, nclasses, allocator, stream, nullptr, nullptr,
      cluster_std, shuffle, center_box_min, center_box_max, seed);
    if (!isRowMajor) {
      D alpha = (D)1.0, beta = (D)0.0;
      MLCommon::LinAlg::cublasgeam<D>(
        handle.getImpl().getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nrows,
        ncols, &alpha, tmpX, ncols, &beta, X, nrows, X, nrows, stream);
      allocator->deallocate(tmpX, nrows * ncols * sizeof(D), stream);
    }
  }

  /**
   * @brief Read the input csv file and construct the dataset
   * @tparam Lambda lambda to customize how to read the data and labels
   * @param handle cuml handle
   * @param csvfile the csv file
   * @param rows number of rows
   * @param cols number of columns
   * @param isRowMajor whether to store the input in row or col major
   * @param nclass number of classes (meaningful only for classification)
   */
  template <typename Lambda>
  void read_csv(const cumlHandle& handle, const std::string& csvfile, int rows,
                int cols, bool isRowMajor, int nclass, Lambda readOp) {
    if (isClassification() && nclass <= 0) {
      ASSERT(false,
             "read_csv: for classification data 'nclasses' is mandatory!");
    }
    nrows = rows;
    ncols = cols;
    nclasses = nclass;
    rowMajor = isRowMajor;
    std::vector<D> _X(nrows * ncols);
    std::vector<L> _y(nrows);
    std::ifstream myfile;
    myfile.open(csvfile);
    std::string line;
    int counter = 0;
    int break_cnt = nrows;
    while (getline(myfile, line) && (counter < nrows)) {
      std::stringstream str(line);
      std::vector<D> row;
      float i;
      while (str >> i) {
        row.push_back(i);
        if (str.peek() == ',') str.ignore();
      }
      readOp(row, _X, _y, counter, rowMajor);
      counter++;
    }
    myfile.close();
    allocate(handle, rows, cols);
    auto stream = handle.getStream();
    MLCommon::copy(X, &(_X[0]), nrows * ncols, stream);
    MLCommon::copy(y, &(_y[0]), nrows, stream);
  }

 private:
  void allocate(const cumlHandle& handle, int rows, int cols) {
    auto allocator = handle.getDeviceAllocator();
    auto stream = handle.getStream();
    nrows = rows;
    ncols = cols;
    X = (D*)allocator->allocate(nrows * ncols * sizeof(D), stream);
    y = (L*)allocator->allocate(nrows * sizeof(L), stream);
  }
};

}  // end namespace Bench
}  // end namespace ML
