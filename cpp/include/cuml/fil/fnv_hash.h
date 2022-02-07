/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <limits.h>

#include <cstdint>
#include <numeric>

// Implements https://tools.ietf.org/html/draft-eastlake-fnv-17.html
// Algorithm is public domain, non-cryptographic strength and no patents or rights to patent.
// If input elements are not 8-bit, such a computation does not match
// the FNV spec.
template <typename It>
unsigned long long fowler_noll_vo_fingerprint64(It begin, It end)
{
  static_assert(sizeof(*begin) == 1, "FNV deals with byte-sized (octet) input arrays only");
  return std::accumulate(
    begin, end, 14695981039346656037ull, [](const unsigned long long& fingerprint, auto x) {
      return (fingerprint * 0x100000001b3ull) ^ x;
    });
}

// xor-folded fingerprint64 to ensure first bits are affected by other input bits
// should give a 1% collision probability within a 10'000 hash set
template <typename It>
uint32_t fowler_noll_vo_fingerprint64_32(It begin, It end)
{
  unsigned long long fp64 = fowler_noll_vo_fingerprint64(begin, end);
  return (fp64 & UINT_MAX) ^ (fp64 >> 32);
}
