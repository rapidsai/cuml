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

#include <chrono>
#include <cuml/common/utils.hpp>
#include <iomanip>
#include <iostream>

namespace MLCommon {

/** helper method to get max usable shared mem per block parameter */
inline int getSharedMemPerBlock() {
  int devId;
  CUDA_CHECK(cudaGetDevice(&devId));
  int smemPerBlk;
  CUDA_CHECK(cudaDeviceGetAttribute(&smemPerBlk,
                                    cudaDevAttrMaxSharedMemoryPerBlock, devId));
  return smemPerBlk;
}
/** helper method to get multi-processor count parameter */
inline int getMultiProcessorCount() {
  int devId;
  CUDA_CHECK(cudaGetDevice(&devId));
  int mpCount;
  CUDA_CHECK(
    cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, devId));
  return mpCount;
}

/**
 * @brief Generic copy method for all kinds of transfers
 * @tparam Type data type
 * @param dst destination pointer
 * @param src source pointer
 * @param len lenth of the src/dst buffers in terms of number of elements
 * @param stream cuda stream
 */
template <typename Type>
void copy(Type* dst, const Type* src, size_t len, cudaStream_t stream) {
  CUDA_CHECK(
    cudaMemcpyAsync(dst, src, len * sizeof(Type), cudaMemcpyDefault, stream));
}

/**
 * @defgroup Copy Copy methods
 * These are here along with the generic 'copy' method in order to improve
 * code readability using explicitly specified function names
 * @{
 */
/** performs a host to device copy */
template <typename Type>
void updateDevice(Type* dPtr, const Type* hPtr, size_t len,
                  cudaStream_t stream) {
  copy(dPtr, hPtr, len, stream);
}

/** performs a device to host copy */
template <typename Type>
void updateHost(Type* hPtr, const Type* dPtr, size_t len, cudaStream_t stream) {
  copy(hPtr, dPtr, len, stream);
}

template <typename Type>
void copyAsync(Type* dPtr1, const Type* dPtr2, size_t len,
               cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(dPtr1, dPtr2, len * sizeof(Type),
                             cudaMemcpyDeviceToDevice, stream));
}

/** helper method to convert an array on device to a string on host */
template <typename T>
std::string arr2Str(const T* arr, int size, std::string name,
                    cudaStream_t stream, int width = 4) {
  std::stringstream ss;

  T* arr_h = (T*)malloc(size * sizeof(T));
  updateHost(arr_h, arr, size, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  ss << name << " = [ ";
  for (int i = 0; i < size; i++) {
    ss << std::setw(width) << arr_h[i];

    if (i < size - 1) ss << ", ";
  }
  ss << " ]" << std::endl;

  free(arr_h);

  return ss.str();
}
/** this seems to be unused, but may be useful in the future */
template <typename T>
void ASSERT_DEVICE_MEM(T* ptr, std::string name) {
  cudaPointerAttributes s_att;
  cudaError_t s_err = cudaPointerGetAttributes(&s_att, ptr);

  if (s_err != 0 || s_att.device == -1)
    std::cout << "Invalid device pointer encountered in " << name
              << ". device=" << s_att.device << ", err=" << s_err << std::endl;
};

inline uint32_t curTimeMillis() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
    .count();
}

/** @} */

/** Helper function to calculate need memory for allocate to store dense matrix.
* @param rows number of rows in matrix
* @param columns number of columns in matrix
* @return need number of items to allocate via allocate()
* @sa allocate()
*/
inline size_t allocLengthForMatrix(size_t rows, size_t columns) {
  return rows * columns;
}

/** cuda malloc */
template <typename Type>
void allocate(Type*& ptr, size_t len, bool setZero = false) {
  CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(Type) * len));
  if (setZero) CUDA_CHECK(cudaMemset(ptr, 0, sizeof(Type) * len));
}

/** Helper function to check alignment of pointer.
* @param ptr the pointer to check
* @param alignment to be checked for
* @return true if address in bytes is a multiple of alignment
*/
template <typename Type>
bool is_aligned(Type* ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

/** calculate greatest common divisor of two numbers
* @a integer
* @b integer
* @ return gcd of a and b
*/
template <typename IntType>
IntType gcd(IntType a, IntType b) {
  while (b != 0) {
    IntType tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

/**
 * @defgroup Debug utils for debug device code
 * @{
 */
template <class T, class OutStream>
void myPrintHostVector(const char* variableName, const T* hostMem,
                       size_t componentsCount, OutStream& out) {
  out << variableName << "=[";
  for (size_t i = 0; i < componentsCount; ++i) {
    if (i != 0) out << ",";
    out << hostMem[i];
  }
  out << "];\n";
}

template <class T>
void myPrintHostVector(const char* variableName, const T* hostMem,
                       size_t componentsCount) {
  myPrintHostVector(variableName, hostMem, componentsCount, std::cout);
  std::cout.flush();
}

template <class T, class OutStream>
void myPrintDevVector(const char* variableName, const T* devMem,
                      size_t componentsCount, OutStream& out) {
  T* hostMem = new T[componentsCount];
  CUDA_CHECK(cudaMemcpy(hostMem, devMem, componentsCount * sizeof(T),
                        cudaMemcpyDeviceToHost));
  myPrintHostVector(variableName, hostMem, componentsCount, out);
  delete[] hostMem;
}

template <class T>
void myPrintDevVector(const char* variableName, const T* devMem,
                      size_t componentsCount) {
  myPrintDevVector(variableName, devMem, componentsCount, std::cout);
  std::cout.flush();
}
/** @} */

};  // end namespace MLCommon
