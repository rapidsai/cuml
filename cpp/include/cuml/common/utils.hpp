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
};  // namespace MLCommon
