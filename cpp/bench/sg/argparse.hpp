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

#include <sys/time.h>
#include <algorithm>
#include <cstdio>
#include <string>

namespace ML {
namespace Bench {

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg,
             const T default_val) {
  T argval = default_val;
  char** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

inline bool get_argval(char** begin, char** end, const std::string& arg) {
  char** itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

#define TIC(start)              \
  do {                          \
    gettimeofday(&start, NULL); \
  } while (0)

#define TOC(start, msg)                                     \
  do {                                                      \
    struct timeval stop;                                    \
    gettimeofday(&stop, NULL);                              \
    double elapsed = (stop.tv_sec - start.tv_sec) * 1000.0; \
    elapsed += (stop.tv_usec - start.tv_usec) / 1000.0;     \
    std::printf("TIMING: %s -> %lfms\n", msg, elapsed);     \
  } while (0)

}  // end namespace Bench
}  // end namespace ML
