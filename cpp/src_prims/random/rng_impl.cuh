/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <curand_kernel.h>
#include <stdint.h>
#include <cuda_utils.cuh>

namespace raft {
namespace random {
namespace detail {

/** Philox-based random number generator */
// Courtesy: Jakub Szuppe
struct PhiloxGenerator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed random seed (can be same across all threads)
   * @param subsequence as found in curand docs
   * @param offset as found in curand docs
   */
  DI PhiloxGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    curand_init(seed, subsequence, offset, &state);
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  DI void next(float& ret) { ret = curand_uniform(&(this->state)); }
  DI void next(double& ret) { ret = curand_uniform_double(&(this->state)); }
  DI void next(uint32_t& ret) { ret = curand(&(this->state)); }
  DI void next(uint64_t& ret) {
    uint32_t a, b;
    next(a);
    next(b);
    ret = (uint64_t)a | ((uint64_t)b << 32);
  }
  DI void next(int32_t& ret) {
    uint32_t val;
    next(val);
    ret = int32_t(val & 0x7fffffff);
  }
  DI void next(int64_t& ret) {
    uint64_t val;
    next(val);
    ret = int64_t(val & 0x7fffffffffffffff);
  }
  /** @} */

 private:
  /** the state for RNG */
  curandStatePhilox4_32_10_t state;
};

/** LFSR taps-filter for generating random numbers. */
// Courtesy: Vinay Deshpande
struct TapsGenerator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed the seed (can be same across all threads)
   * @param subsequence unused
   * @param offset unused
   */
  DI TapsGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    uint64_t delta = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    delta += ((blockIdx.y * blockDim.y) + threadIdx.y) * stride;
    stride *= blockDim.y * gridDim.y;
    delta += ((blockIdx.z * blockDim.z) + threadIdx.z) * stride;
    state = seed + delta + 1;
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  template <typename Type>
  DI void next(Type& ret) {
    constexpr double ULL_LARGE = 1.8446744073709551614e19;
    uint64_t val;
    next(val);
    ret = static_cast<Type>(val);
    ret /= static_cast<Type>(ULL_LARGE);
  }
  DI void next(uint64_t& ret) {
    constexpr uint64_t TAPS = 0x8000100040002000ULL;
    constexpr int ROUNDS = 128;
    for (int i = 0; i < ROUNDS; i++)
      state = (state >> 1) ^ (-(state & 1ULL) & TAPS);
    ret = state;
  }
  DI void next(uint32_t& ret) {
    uint64_t val;
    next(val);
    ret = (uint32_t)val;
  }
  DI void next(int32_t& ret) {
    uint32_t val;
    next(val);
    ret = int32_t(val & 0x7fffffff);
  }
  DI void next(int64_t& ret) {
    uint64_t val;
    next(val);
    ret = int64_t(val & 0x7fffffffffffffff);
  }
  /** @} */

 private:
  /** the state for RNG */
  uint64_t state;
};

/** Kiss99-based random number generator */

struct Kiss99Generator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed the seed (can be same across all threads)
   * @param subsequence unused
   * @param offset unused
   */
  DI Kiss99Generator(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    initKiss99(seed);
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  template <typename Type>
  DI void next(Type& ret) {
    constexpr double U_LARGE = 4.294967295e9;
    uint32_t val;
    next(val);
    ret = static_cast<Type>(val);
    ret /= static_cast<Type>(U_LARGE);
  }
  DI void next(uint32_t& ret) {
    uint32_t MWC;
    z = 36969 * (z & 65535) + (z >> 16);
    w = 18000 * (w & 65535) + (w >> 16);
    MWC = ((z << 16) + w);
    jsr ^= (jsr << 17);
    jsr ^= (jsr >> 13);
    jsr ^= (jsr << 5);
    jcong = 69069 * jcong + 1234567;
    MWC = ((MWC ^ jcong) + jsr);
    ret = MWC;
  }
  DI void next(uint64_t& ret) {
    uint32_t a, b;
    next(a);
    next(b);
    ret = (uint64_t)a | ((uint64_t)b << 32);
  }
  DI void next(int32_t& ret) {
    uint32_t val;
    next(val);
    ret = int32_t(val & 0x7fffffff);
  }
  DI void next(int64_t& ret) {
    uint64_t val;
    next(val);
    ret = int64_t(val & 0x7fffffffffffffff);
  }
  /** @} */

 private:
  /** one of the kiss99 states */
  uint32_t z;
  /** one of the kiss99 states */
  uint32_t w;
  /** one of the kiss99 states */
  uint32_t jsr;
  /** one of the kiss99 states */
  uint32_t jcong;

  // This function multiplies 128-bit hash by 128-bit FNV prime and returns lower
  // 128 bits. It uses 32-bit wide multiply only.
  DI void mulByFnv1a128Prime(uint32_t* h) {
    typedef union {
      uint32_t u32[2];
      uint64_t u64[1];
    } words64;

    // 128-bit FNV prime = p3 * 2^96 + p2 * 2^64 + p1 * 2^32 + p0
    // Here p0 = 315, p2 = 16777216, p1 = p3 = 0
    const uint32_t p0 = uint32_t(315), p2 = uint32_t(16777216);
    // Partial products
    words64 h0p0, h1p0, h2p0, h0p2, h3p0, h1p2;

    h0p0.u64[0] = uint64_t(h[0]) * p0;
    h1p0.u64[0] = uint64_t(h[1]) * p0;
    h2p0.u64[0] = uint64_t(h[2]) * p0;
    h0p2.u64[0] = uint64_t(h[0]) * p2;
    h3p0.u64[0] = uint64_t(h[3]) * p0;
    h1p2.u64[0] = uint64_t(h[1]) * p2;

    // h_n[0] = LO(h[0]*p[0]);
    // h_n[1] = HI(h[0]*p[0]) + LO(h[1]*p[0]);
    // h_n[2] = HI(h[1]*p[0]) + LO(h[2]*p[0]) + LO(h[0]*p[2]);
    // h_n[3] = HI(h[2]*p[0]) + HI(h[0]*p[2]) + LO(h[3]*p[0]) + LO(h[1]*p[2]);
    uint32_t carry = 0;
    h[0] = h0p0.u32[0];

    h[1] = h0p0.u32[1] + h1p0.u32[0];
    carry = h[1] < h0p0.u32[1] ? 1 : 0;

    h[2] = h1p0.u32[1] + carry;
    carry = h[2] < h1p0.u32[1] ? 1 : 0;
    h[2] += h2p0.u32[0];
    carry = h[2] < h2p0.u32[0] ? carry + 1 : carry;
    h[2] += h0p2.u32[0];
    carry = h[2] < h0p2.u32[0] ? carry + 1 : carry;

    h[3] = h2p0.u32[1] + h0p2.u32[1] + h3p0.u32[0] + h1p2.u32[0] + carry;
    return;
  }

  DI void fnv1a128(uint32_t* hash, uint32_t txt) {
    hash[0] ^= (txt >> 0) & 0xFF;
    mulByFnv1a128Prime(hash);
    hash[0] ^= (txt >> 8) & 0xFF;
    mulByFnv1a128Prime(hash);
    hash[0] ^= (txt >> 16) & 0xFF;
    mulByFnv1a128Prime(hash);
    hash[0] ^= (txt >> 24) & 0xFF;
    mulByFnv1a128Prime(hash);
  }

  DI void initKiss99(uint64_t seed) {
    // Initialize hash to 128-bit FNV1a basis
    uint32_t hash[4] = {1653982605UL, 1656234357UL, 129696066UL, 1818371886UL};

    // Digest threadIdx, blockIdx and seed
    fnv1a128(hash, threadIdx.x);
    fnv1a128(hash, threadIdx.y);
    fnv1a128(hash, threadIdx.z);
    fnv1a128(hash, blockIdx.x);
    fnv1a128(hash, blockIdx.y);
    fnv1a128(hash, blockIdx.z);
    fnv1a128(hash, uint32_t(seed));
    fnv1a128(hash, uint32_t(seed >> 32));

    // Initialize KISS99 state with hash
    z = hash[0];
    w = hash[1];
    jsr = hash[2];
    jcong = hash[3];
  }
};

/**
 * @brief generator-agnostic way of generating random numbers
 * @tparam GenType the generator object that expose 'next' method
 */
template <typename GenType>
struct Generator {
  DI Generator(uint64_t seed, uint64_t subsequence, uint64_t offset)
    : gen(seed, subsequence, offset) {}

  template <typename Type>
  DI void next(Type& ret) {
    gen.next(ret);
  }

 private:
  /** the actual generator */
  GenType gen;
};

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
