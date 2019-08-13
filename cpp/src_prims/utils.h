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

#include <cuda_runtime.h>
#include <execinfo.h>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace MLCommon {

/** base exception class for the cuML or ml-prims project */
class Exception : public std::exception {
 public:
  /** default ctor */
  Exception() throw() : std::exception(), msg() {}

  /** copy ctor */
  Exception(const Exception& src) throw() : std::exception(), msg(src.what()) {
    collectCallStack();
  }

  /** ctor from an input message */
  Exception(const std::string& _msg) throw() : std::exception(), msg(_msg) {
    collectCallStack();
  }

  /** dtor */
  virtual ~Exception() throw() {}

  /** get the message associated with this exception */
  virtual const char* what() const throw() { return msg.c_str(); }

 private:
  /** message associated with this exception */
  std::string msg;

  /** append call stack info to this exception's message for ease of debug */
  // Courtesy: https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
  void collectCallStack() throw() {
#ifdef __GNUC__
    const int MaxStackDepth = 64;
    void* stack[MaxStackDepth];
    auto depth = backtrace(stack, MaxStackDepth);
    std::ostringstream oss;
    oss << std::endl << "Obtained " << depth << " stack frames" << std::endl;
    char** strings = backtrace_symbols(stack, depth);
    if (strings == nullptr) {
      oss << "But no stack trace could be found!" << std::endl;
      msg += oss.str();
      return;
    }
    ///@todo: support for demangling of C++ symbol names
    for (int i = 0; i < depth; ++i) {
      oss << "#" << i << " in " << strings[i] << std::endl;
    }
    free(strings);
    msg += oss.str();
#endif  // __GNUC__
  }
};

/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                    \
  do {                                                                     \
    std::string msg;                                                       \
    char errMsg[2048];                                                     \
    std::sprintf(errMsg, "Exception occured! file=%s line=%d: ", __FILE__, \
                 __LINE__);                                                \
    msg += errMsg;                                                         \
    std::sprintf(errMsg, fmt, ##__VA_ARGS__);                              \
    msg += errMsg;                                                         \
    throw MLCommon::Exception(msg);                                        \
  } while (0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)              \
  do {                                       \
    if (!(check)) THROW(fmt, ##__VA_ARGS__); \
  } while (0)

/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t status = call;                                           \
    ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s\n", #call, \
           cudaGetErrorString(status));                                  \
  } while (0)

/** check for cuda runtime API errors but log error instead of raising
 *  exception.
 *  @todo: This will need to use our common logging infrastructure once
 *  that is in place.
 */
#define CUDA_CHECK_NO_THROW(call)                                              \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr,                                                     \
                   "ERROR: CUDA call='%s' at file=%s line=%d failed with %s ", \
                   #call, __FILE__, __LINE__, cudaGetErrorString(status));     \
    }                                                                          \
  } while (0)

/** helper method to get max usable shared mem per block parameter */
inline int getSharedMemPerBlock() {
  int devId;
  CUDA_CHECK(cudaGetDevice(&devId));
  int smemPerBlk;
  CUDA_CHECK(cudaDeviceGetAttribute(&smemPerBlk,
                                    cudaDevAttrMaxSharedMemoryPerBlock, devId));
  return smemPerBlk;
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
