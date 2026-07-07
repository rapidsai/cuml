/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/tree/algo_helper.h>

#include <cstdint>

namespace ML {
namespace DT {

// 32-bit FNV1a hash
// Reference: http://www.isthe.com/chongo/tech/comp/fnv/index.html
constexpr uint32_t fnv1a32_prime = uint32_t(16777619);
constexpr uint32_t fnv1a32_basis = uint32_t(2166136261);

HDI uint32_t fnv1a32(uint32_t hash, uint32_t txt)
{
  hash ^= (txt >> 0) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 8) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 16) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 24) & 0xFF;
  hash *= fnv1a32_prime;
  return hash;
}

template <typename T>
HDI uint32_t fnv1a32_combine(uint32_t hash, T value)
{
  hash = fnv1a32(hash, static_cast<uint32_t>(value));
  if constexpr (sizeof(T) > sizeof(uint32_t)) {
    hash = fnv1a32(hash, static_cast<uint32_t>(static_cast<uint64_t>(value) >> 32));
  }
  return hash;
}

template <typename... Ts>
HDI uint32_t fnv1a32_hash(Ts... values)
{
  uint32_t hash = fnv1a32_basis;
  ((hash = fnv1a32_combine(hash, values)), ...);
  return hash;
}

}  // namespace DT
}  // namespace ML
