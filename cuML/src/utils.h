/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <stdexcept>
#include <string>
#include <cstdio>

namespace Dbscan {

#define THROW(fmt, ...)                                                 \
    do {                                                                \
        std::string msg;                                                \
        char errMsg[2048];                                              \
        std::sprintf(errMsg, "Exception occured! file=%s line=%d: ",    \
                     __FILE__, __LINE__);                               \
        msg += errMsg;                                                  \
        std::sprintf(errMsg, fmt, ##__VA_ARGS__);                       \
        msg += errMsg;                                                  \
        throw std::runtime_error(msg);                                  \
    } while(0)

#define ASSERT(check, fmt, ...)                  \
    do {                                         \
        if(!(check))  THROW(fmt, ##__VA_ARGS__); \
    } while(0)


inline int ceildiv(int a, int b) {
    return ((a + b - 1) / b);
}
inline int alignSize(int a, int b) {
    return ceildiv(a,b)*b;
}

} // namespace Dbscan
