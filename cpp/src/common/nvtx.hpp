/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>

namespace ML {

/**
 * @brief Synchronize CUDA stream and push a named nvtx range
 * @param name range name
 * @param stream stream to synchronize
 */
void PUSH_RANGE(const char* name, cudaStream_t stream);

/**
 * @brief Synchronize CUDA stream and pop the latest nvtx range
 * @param stream stream to synchronize
 */
void POP_RANGE(cudaStream_t stream);

/**
 * @brief Push a named nvtx range
 * @param name range name
 */
void PUSH_RANGE(const char* name);

/** Pop the latest range */
void POP_RANGE();

/** Push a named nvtx range that would be popped at the end of the object lifetime. */
class AUTO_RANGE {
 private:
  std::optional<rmm::cuda_stream_view> stream;

  template <typename... Args>
  void init(const char* name, Args... args)
  {
    if constexpr (sizeof...(args) > 0) {
      int length = std::snprintf(nullptr, 0, name, args...);
      assert(length >= 0);
      auto buf = std::make_unique<char[]>(length + 1);
      std::snprintf(buf.get(), length + 1, name, args...);

      if (stream.has_value())
        PUSH_RANGE(buf.get(), stream.value());
      else
        PUSH_RANGE(buf.get());
    } else {
      if (stream.has_value())
        PUSH_RANGE(name, stream.value());
      else
        PUSH_RANGE(name);
    }
  }

 public:
  /**
   * Synchronize CUDA stream and push a named nvtx range
   * At the end of the object lifetime, synchronize again and pop the range.
   *
   * @param stream stream to synchronize
   * @param name range name (accepts printf-style arguments)
   * @param args the arguments for the printf-style formatting
   */
  template <typename... Args>
  AUTO_RANGE(rmm::cuda_stream_view stream, const char* name, Args... args)
    : stream(std::make_optional(stream))
  {
    init(name, args...);
  }

  /**
   * Push a named nvtx range.
   * At the end of the object lifetime, pop the range back.
   *
   * @param name range name (accepts printf-style arguments)
   * @param args the arguments for the printf-style formatting
   */
  template <typename... Args>
  AUTO_RANGE(const char* name, Args... args) : stream(std::nullopt)
  {
    init(name, args...);
  }

  ~AUTO_RANGE()
  {
    if (stream.has_value())
      POP_RANGE(stream.value());
    else
      POP_RANGE();
  }
};

/*!
  \def CUML_USING_RANGE(...)
  When NVTX is enabled, push a named nvtx range and pop it at the end of the enclosing code block.

  This macro initializes a dummy AUTO_RANGE variable on the stack,
  which pushes the range in its constructor and pops it in the destructor.
*/
#ifdef NVTX_ENABLED
#define CUML_USING_RANGE(...) ML::AUTO_RANGE _AUTO_RANGE_##__LINE__(__VA_ARGS__)
#else
#define CUML_USING_RANGE(...) (void)0
#endif

}  // end namespace ML
