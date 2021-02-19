/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.; 
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,;
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#ifndef WHTEST
#define WHTEST
#include <raft/cudart_utils.h>

namespace raft {
extern template void copy<float>(float*, const float*, size_t, cudaStream_t);
extern template void copy<double>(double*, const double*, size_t, cudaStream_t);
extern template void copy<int>(int*, const int*, size_t, cudaStream_t);
extern template void copy<char>(char*, const char*, size_t, cudaStream_t);
extern template void copy<long>(long*, const long*, size_t, cudaStream_t);
extern template void copy<bool>(bool*, const bool*, size_t, cudaStream_t);
extern template void copy<unsigned long>(unsigned long*, const unsigned long*,
                                         size_t, cudaStream_t);
extern template void copy<unsigned int>(unsigned int*, const unsigned int*,
                                        size_t, cudaStream_t);
extern template void copy<unsigned long long>(unsigned long long*,
                                              const unsigned long long*, size_t,
                                              cudaStream_t);
}  // namespace raft
#endif
