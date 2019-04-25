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

#include <cstdio>
#include <execinfo.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

namespace MLCommon {

/** base exception class for the cuML or ml-prims project */
class Exception : public std::exception {
public:
  /** default ctor */
  Exception() throw(): std::exception(), msg() {}

  /** copy ctor */
  Exception(const Exception& src) throw(): std::exception(), msg(src.what()) {
    collectCallStack();
  }

  /** ctor from an input message */
  Exception(const std::string& _msg) throw(): std::exception(), msg(_msg) {
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
#endif // __GNUC__
  }
};

/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                        \
  do {                                                                         \
    std::string msg;                                                           \
    char errMsg[2048];                                                         \
    std::sprintf(errMsg, "Exception occured! file=%s line=%d: ", __FILE__,     \
                 __LINE__);                                                    \
    msg += errMsg;                                                             \
    std::sprintf(errMsg, fmt, ##__VA_ARGS__);                                  \
    msg += errMsg;                                                             \
    throw MLCommon::Exception(msg);                                            \
  } while (0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)                                                \
  do {                                                                         \
    if (!(check))                                                              \
      THROW(fmt, ##__VA_ARGS__);                                               \
  } while (0)

/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s\n", #call,       \
           cudaGetErrorString(status));                                        \
  } while (0)



///@todo: add a similar CUDA_CHECK_NO_THROW
/// (Ref: https://github.com/rapidsai/cuml/issues/229)

/**
 * @defgroup Copy Copy methods
 * @{
 */
/** performs a host to device copy */
template <typename Type>
void updateDevice(Type *dPtr, const Type *hPtr, size_t len,
                  cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpy(dPtr, hPtr, len * sizeof(Type),
                        cudaMemcpyHostToDevice));
}

/** performs an sync host to device copy */
template <typename Type>
void updateDeviceAsync(Type *dPtr, const Type *hPtr, size_t len,
                       cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(dPtr, hPtr, len * sizeof(Type),
                             cudaMemcpyHostToDevice, stream));
}

/** performs a device to host copy */
template <typename Type>
void updateHost(Type *hPtr, const Type *dPtr, size_t len,
                cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpy(hPtr, dPtr, len * sizeof(Type),
                        cudaMemcpyDeviceToHost));
}

/** performs an async device to host copy */
template <typename Type>
void updateHostAsync(Type *hPtr, const Type *dPtr, size_t len,
                     cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(hPtr, dPtr, len * sizeof(Type),
                             cudaMemcpyDeviceToHost, stream));
}

/** performs a device to device copy */
template <typename Type>
void copy(Type* dPtr1, const Type* dPtr2, size_t len) {
  CUDA_CHECK(cudaMemcpy(dPtr1, dPtr2, len*sizeof(Type),
                        cudaMemcpyDeviceToDevice));
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
void allocate(Type *&ptr, size_t len, bool setZero = false) {
  CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * len));
  if (setZero)
    CUDA_CHECK(cudaMemset(ptr, 0, sizeof(Type) * len));
}

/** Helper function to check alignment of pointer.
* @param ptr the pointer to check
* @param alignment to be checked for
* @return true if address in bytes is a multiple of alignment
*/
template <typename Type>
bool is_aligned(Type *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

/** calculate greatest common divisor of two numbers
* @a integer
* @b integer
* @ return gcd of a and b
*/
template <typename IntType>
IntType gcd(IntType a, IntType b) {
    while(b!=0) {
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
template<class T, class OutStream>
void myPrintHostVector(const char * variableName, const T * hostMem, size_t componentsCount, OutStream& out)
{
    out << variableName << "=[";
    for (size_t i = 0; i < componentsCount; ++i)
    {
        if (i != 0)
            out << ",";
        out << hostMem[i];
    }
    out << "];\n";
}

template<class T>
void myPrintHostVector(const char * variableName, const T * hostMem, size_t componentsCount)
{
    myPrintHostVector(variableName, hostMem, componentsCount, std::cout);
    std::cout.flush();
}

template<class T, class OutStream>
void myPrintDevVector(const char * variableName, const T * devMem, size_t componentsCount, OutStream& out)
{
    T* hostMem = new T[componentsCount];
    CUDA_CHECK(cudaMemcpy(hostMem, devMem, componentsCount * sizeof(T), cudaMemcpyDeviceToHost));
    myPrintHostVector(variableName, hostMem, componentsCount, out);
    delete []hostMem;
}

template<class T>
void myPrintDevVector(const char * variableName, const T * devMem, size_t componentsCount)
{
    myPrintDevVector(variableName, devMem, componentsCount, std::cout);
    std::cout.flush();
}
/** @} */

}; // end namespace MLCommon
