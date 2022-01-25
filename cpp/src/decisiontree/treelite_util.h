/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cstdint>

namespace ML {
namespace DT {

template <typename T>
class TreeliteType;

template <>
class TreeliteType<float> {
 public:
  static constexpr const char* value = "float32";
};

template <>
class TreeliteType<double> {
 public:
  static constexpr const char* value = "float64";
};

template <>
class TreeliteType<uint32_t> {
 public:
  static constexpr const char* value = "uint32";
};

template <>
class TreeliteType<int> {
 public:
  static_assert(sizeof(int) == sizeof(uint32_t), "int must be 32-bit");
  static constexpr const char* value = "uint32";
};

}  // End namespace DT

}  // End namespace ML
