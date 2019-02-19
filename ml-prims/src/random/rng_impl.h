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

#include <curand_kernel.h>
#include <stdint.h>
#include "cuda_utils.h"


namespace MLCommon {
namespace Random {
namespace detail {

/**
 * @defgroup PhiloxGenerator Philox-based random number generator
 * @{
 */
// Courtesy: Jakub Szuppe
/** base class for philox based RNG's */
struct PhiloxGeneratorBase {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed random seed (can be same across all threads)
   * @param subsequence as found in curand docs
   * @param offset as found in curand docs
   */
  DI PhiloxGeneratorBase(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    curand_init(seed, subsequence, offset, &state);
  }

protected:
  /** the state for RNG */
  curandStatePhilox4_32_10_t state;
};

template <class T>
struct PhiloxGenerator;

template <>
struct PhiloxGenerator<float> : public PhiloxGeneratorBase {
  using PhiloxGeneratorBase::PhiloxGeneratorBase;

  DI float next() { return curand_uniform(&(this->state)); }

  typedef float MathType;
};

template <>
struct PhiloxGenerator<double> : public PhiloxGeneratorBase {
  using PhiloxGeneratorBase::PhiloxGeneratorBase;

  DI double next() { return curand_uniform_double(&(this->state)); }

  typedef double MathType;
};
/** @} */


/**
 * @defgroup TapsGenerator Taps-based random number generator
 * @{
 */
// Courtesy: Vinay Deshpande
/**
 * @brief LFSR taps-filter for generating random numbers. This is SUPER-slow,
 * but in my experience, generates "better quality" random numbers.
 */
template <typename Type>
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

  /** generate the next uniform random number between 0.0 and 1.0 */
  DI Type next() {
    constexpr uint64_t TAPS = 0x8000100040002000ULL;
    constexpr int ROUNDS = 128;
    constexpr double ULL_LARGE = 1.8446744073709551614e19;
    for (int i = 0; i < ROUNDS; i++)
      state = (state >> 1) ^ (-(state & 1ULL) & TAPS);
    Type res = static_cast<Type>(state);
    res /= static_cast<Type>(ULL_LARGE);
    return res;
  }

  typedef Type MathType;

private:
  /** the state for RNG */
  uint64_t state;
};
/** @} */


/**
 * @defgroup Kiss99Generator Kiss99-based random number generator
 * @{
 */
// Courtesy: Vinay Deshpande
template <typename Type>
struct Kiss99Generator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed the seed (can be same across all threads)
   * @param subsequence unused
   * @param offset unused
   */
  DI Kiss99Generator(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    initKiss99((uint32_t)seed);
  }

  /** generate the next uniform random number between 0.0 and 1.0 */
  DI Type next() {
    constexpr double U_LARGE = 4.294967295e9;
    uint32_t MWC;
    z = 36969 * (z & 65535) + (z >> 16);
    w = 18000 * (w & 65535) + (w >> 16);
    MWC = ((z << 16) + w);
    jsr ^= (jsr << 17);
    jsr ^= (jsr >> 13);
    jsr ^= (jsr << 5);
    jcong = 69069 * jcong + 1234567;
    MWC = ((MWC ^ jcong) + jsr);
    Type res = static_cast<Type>(MWC);
    res /= static_cast<Type>(U_LARGE);
    return res;
  }

  typedef Type MathType;

private:
  /** one of the kiss99 states */
  uint32_t z;
  /** one of the kiss99 states */
  uint32_t w;
  /** one of the kiss99 states */
  uint32_t jsr;
  /** one of the kiss99 states */
  uint32_t jcong;

  static const uint32_t fnvBasis = 2166136261U;
  static const uint32_t fnvPrime = 16777619U;

  DI void fnv1a32(uint32_t &hash, uint32_t txt) {
    hash ^= (txt >> 0) & 0xFF;
    hash *= fnvPrime;
    hash ^= (txt >> 8) & 0xFF;
    hash *= fnvPrime;
    hash ^= (txt >> 16) & 0xFF;
    hash *= fnvPrime;
    hash ^= (txt >> 24) & 0xFF;
    hash *= fnvPrime;
  }

  DI void initKiss99(uint32_t seed) {
    uint32_t hash = fnvBasis;
    fnv1a32(hash, uint32_t(threadIdx.x));
    fnv1a32(hash, uint32_t(threadIdx.y));
    fnv1a32(hash, uint32_t(threadIdx.z));
    fnv1a32(hash, uint32_t(blockIdx.x));
    fnv1a32(hash, uint32_t(blockIdx.y));
    fnv1a32(hash, uint32_t(blockIdx.z));
    fnv1a32(hash, seed);
    z = hash;
    fnv1a32(hash, 0x01);
    w = hash;
    fnv1a32(hash, 0x01);
    jsr = hash;
    fnv1a32(hash, 0x01);
    jcong = hash;
  }
};
/** @} */


/**
 * @brief generator-agnostic way of generating random numbers
 * @tparam GenType the generator object that expose 'next' method
 */
template <typename GenType>
struct Generator {
  typedef typename GenType::MathType MathType;

  DI Generator(uint64_t seed, uint64_t subsequence, uint64_t offset)
    : gen(seed, subsequence, offset) {}

  DI MathType next() { return gen.next(); }

private:
  /** the actual generator */
  GenType gen;
};

}; // end namespace detail
}; // end namespace Random
}; // end namespace MLCommon
