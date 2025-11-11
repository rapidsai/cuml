/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
