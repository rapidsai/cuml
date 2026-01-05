/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace MLCommon {
namespace Matrix {

/**
 * @brief This is a *helper* wrapper around the multi-gpu data blocks owned
 * by a worker. It's design is NOT final. Its so written this way to get
 * something concrete in a short span of time.
 * @todo add support for custom allocators
 */
template <typename Type>
struct Data {
  Data() : ptr(nullptr), totalSize(0) {}
  Data(Type* _ptr, size_t _n_elements) : ptr(_ptr), totalSize(_n_elements * sizeof(Type)) {}

  /**
   * actual data block. This is just a linearly laid out buffer of all blocks
   * owned by this worker
   */
  Type* ptr = nullptr;

  /**
   * total size (in bytes) of this buffer. In future, this will be passed
   * to the dealloc function underneath
   */
  size_t totalSize = (size_t)0;

  /**
   * Return the number of elements of Type in ptr.
   */
  size_t numElements() const { return totalSize / sizeof(Type); }
};

typedef Data<float> floatData_t;
typedef Data<double> doubleData_t;

};  // end namespace Matrix
};  // end namespace MLCommon
