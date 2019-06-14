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

#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "cuda_utils.h"
#include "rng_impl.h"

namespace MLCommon {
namespace Random {

/** all different generator types used */
enum GeneratorType {
  /** curand-based philox generator */
  GenPhilox = 0,
  /** LFSR taps generator */
  GenTaps,
  /** kiss99 generator (currently the fastest) */
  GenKiss99
};

inline uint64_t _nextSeed() {
  // because rand() has poor randomness in lower 16b
  uint64_t t0 = (uint64_t)(rand() & 0xFFFF0000) >> 16;
  uint64_t t1 = (uint64_t)(rand() & 0xFFFF0000);
  uint64_t t2 = (uint64_t)(rand() & 0xFFFF0000) >> 16;
  uint64_t t3 = (uint64_t)(rand() & 0xFFFF0000);
  return t0 | t1 | t2 | t3;
}

template <typename OutType, typename MathType, typename GenType,
          typename LenType, typename Lambda>
__global__ void randKernel(uint64_t seed, uint64_t offset, OutType *ptr,
                           LenType len, Lambda randOp) {
  LenType tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  detail::Generator<GenType> gen(seed, (uint64_t)tid, offset);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride) {
    MathType val;
    gen.next(val);
    ptr[idx] = randOp(val, idx);
  }
}

// used for Box-Muller type transformations
template <typename OutType, typename MathType, typename GenType,
          typename LenType, typename Lambda2>
__global__ void rand2Kernel(uint64_t seed, uint64_t offset, OutType *ptr,
                            LenType len, Lambda2 rand2Op) {
  LenType tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  detail::Generator<GenType> gen(seed, (uint64_t)tid, offset);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride) {
    MathType val1, val2;
    gen.next(val1);
    gen.next(val2);
    rand2Op(val1, val2, idx);
    if (idx < len) ptr[idx] = (OutType)val1;
    idx += stride;
    if (idx < len) ptr[idx] = (OutType)val2;
  }
}

template <bool IsNormal, typename Type, typename LenType>
uint64_t _setupSeeds(uint64_t &seed, uint64_t &offset, LenType len,
                     int nThreads, int nBlocks) {
  LenType itemsPerThread = ceildiv(len, LenType(nBlocks * nThreads));
  if (IsNormal && itemsPerThread % 2 == 1) {
    ++itemsPerThread;
  }
  // curand uses 2 32b uint's to generate one double
  uint64_t factor = sizeof(Type) / sizeof(float);
  if (factor == 0) ++factor;
  // Check if there are enough random numbers left in sequence
  // If not, then generate new seed and start from zero offset
  uint64_t newOffset = offset + LenType(itemsPerThread) * factor;
  if (newOffset < offset) {
    offset = 0;
    seed = _nextSeed();
    newOffset = itemsPerThread * factor;
  }
  return newOffset;
}

template <typename OutType, typename MathType = OutType, typename LenType = int,
          typename Lambda>
void randImpl(uint64_t &offset, OutType *ptr, LenType len, Lambda randOp,
              int nThreads, int nBlocks, GeneratorType type,
              cudaStream_t stream) {
  if (len <= 0) return;
  uint64_t seed = _nextSeed();
  auto newOffset =
    _setupSeeds<false, MathType, LenType>(seed, offset, len, nThreads, nBlocks);
  switch (type) {
    case GenPhilox:
      randKernel<OutType, MathType, detail::PhiloxGenerator, LenType, Lambda>
        <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, randOp);
      break;
    case GenTaps:
      randKernel<OutType, MathType, detail::TapsGenerator, LenType, Lambda>
        <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, randOp);
      break;
    case GenKiss99:
      randKernel<OutType, MathType, detail::Kiss99Generator, LenType, Lambda>
        <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, randOp);
      break;
    default:
      ASSERT(false, "randImpl: Incorrect generator type! %d", type);
  };
  CUDA_CHECK(cudaPeekAtLastError());
  offset = newOffset;
}

template <typename OutType, typename MathType = OutType, typename LenType = int,
          typename Lambda2>
void rand2Impl(uint64_t &offset, OutType *ptr, LenType len, Lambda2 rand2Op,
               int nThreads, int nBlocks, GeneratorType type,
               cudaStream_t stream) {
  if (len <= 0) return;
  uint64_t seed = _nextSeed();
  auto newOffset =
    _setupSeeds<true, MathType, LenType>(seed, offset, len, nThreads, nBlocks);
  switch (type) {
    case GenPhilox:
      rand2Kernel<OutType, MathType, detail::PhiloxGenerator, LenType, Lambda2>
        <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, rand2Op);
      break;
    case GenTaps:
      rand2Kernel<OutType, MathType, detail::TapsGenerator, LenType, Lambda2>
        <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, rand2Op);
      break;
    case GenKiss99:
      rand2Kernel<OutType, MathType, detail::Kiss99Generator, LenType, Lambda2>
        <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, rand2Op);
      break;
    default:
      ASSERT(false, "rand2Impl: Incorrect generator type! %d", type);
  };
  CUDA_CHECK(cudaPeekAtLastError());
  offset = newOffset;
}

template <typename Type>
__global__ void constFillKernel(Type *ptr, int len, Type val) {
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned idx = tid; idx < len; idx += stride) {
    ptr[idx] = val;
  }
}

/** The main random number generator class, fully on GPUs */
class Rng {
 public:
  /** ctor */
  Rng(uint64_t _s, GeneratorType _t = GenPhilox) : type(_t) {
    srand(_s);
    offset = 0;
    // simple heuristic to make sure all SMs will be occupied properly
    // and also not too many initialization calls will be made by each thread
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
    nBlocks = 4 * props.multiProcessorCount;
  }

  /**
   * @brief Generate uniformly distributed numbers in the given range
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param start start of the range
   * @param end end of the range
   * @param stream stream where to launch the kernel
   * @{
   */
  template <typename Type, typename LenType = int>
  void uniform(Type *ptr, LenType len, Type start, Type end,
               cudaStream_t stream) {
    static_assert(std::is_floating_point<Type>::value,
                  "Type for 'uniform' can only be floating point type!");
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        return (val * (end - start)) + start;
      },
      NumThreads, nBlocks, type, stream);
  }
  template <typename IntType, typename LenType = int>
  void uniformInt(IntType *ptr, LenType len, IntType start, IntType end,
                  cudaStream_t stream) {
    static_assert(std::is_integral<IntType>::value,
                  "Type for 'uniformInt' can only be integer type!");
    randImpl(
      offset, ptr, len,
      [=] __device__(IntType val, LenType idx) {
        return (val % (end - start)) + start;
      },
      NumThreads, nBlocks, type, stream);
  }
  /** @} */

  /**
   * @brief Generate normal distributed numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void normal(Type *ptr, LenType len, Type mu, Type sigma,
              cudaStream_t stream) {
    rand2Impl(
      offset, ptr, len,
      [=] __device__(Type & val1, Type & val2, LenType idx) {
        constexpr Type twoPi = Type(2.0) * Type(3.141592654);
        constexpr Type minus2 = -Type(2.0);
        Type R = mySqrt(minus2 * myLog(val1));
        Type theta = twoPi * val2;
        Type s, c;
        mySinCos(theta, s, c);
        val1 = R * c * sigma + mu;
        val2 = R * s * sigma + mu;
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Fill an array with the given value
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param val value to be filled
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void fill(Type *ptr, LenType len, Type val, cudaStream_t stream) {
    constFillKernel<Type><<<nBlocks, NumThreads, 0, stream>>>(ptr, len, val);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  /**
   * @brief Generate bernoulli distributed boolean array
   * @tparam Type data type in which to compute the probabilities
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param prob coin-toss probability for heads
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void bernoulli(bool *ptr, LenType len, Type prob, cudaStream_t stream) {
    randImpl<bool, Type>(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) { return val > prob; }, NumThreads,
      nBlocks, type, stream);
  }

  /**
   * @brief Generate bernoulli distributed array and applies scale
   * @tparam Type data type in which to compute the probabilities
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param prob coin-toss probability for heads
   * @param scale scaling factor
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void scaled_bernoulli(Type *ptr, LenType len, Type prob, Type scale,
                        cudaStream_t stream) {
    static_assert(std::is_floating_point<Type>::value,
                  "Type for 'uniform' can only be floating point type!");
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        return val > prob ? -scale : scale;
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate gumbel distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param beta scale value
   * @param stream stream where to launch the kernel
   * @note https://en.wikipedia.org/wiki/Gumbel_distribution
   */
  template <typename Type, typename LenType = int>
  void gumbel(Type *ptr, LenType len, Type mu, Type beta, cudaStream_t stream) {
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        return mu - beta * myLog(-myLog(val));
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate lognormal distributed numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void lognormal(Type *ptr, LenType len, Type mu, Type sigma,
                 cudaStream_t stream) {
    rand2Impl(
      offset, ptr, len,
      [=] __device__(Type & val1, Type & val2, LenType idx) {
        constexpr Type twoPi = Type(2.0) * Type(3.141592654);
        constexpr Type minus2 = -Type(2.0);
        Type R = mySqrt(minus2 * myLog(val1));
        Type theta = twoPi * val2;
        Type s, c;
        mySinCos(theta, s, c);
        val1 = R * c * sigma + mu;
        val2 = R * s * sigma + mu;
        val1 = myExp(val1);
        val2 = myExp(val2);
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate logistic distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param scale scale value
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void logistic(Type *ptr, LenType len, Type mu, Type scale,
                cudaStream_t stream) {
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        return mu - scale * myLog(one / val - one);
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate exponentially distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param lambda the lambda
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void exponential(Type *ptr, LenType len, Type lambda, cudaStream_t stream) {
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        return -myLog(one - val) / lambda;
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate rayleigh distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param sigma the sigma
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void rayleigh(Type *ptr, LenType len, Type sigma, cudaStream_t stream) {
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        constexpr Type two = (Type)2.0;
        return mySqrt(-two * myLog(one - val)) * sigma;
      },
      NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate laplace distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu the mean
   * @param scale the scale
   * @param stream stream where to launch the kernel
   */
  template <typename Type, typename LenType = int>
  void laplace(Type *ptr, LenType len, Type mu, Type scale,
               cudaStream_t stream) {
    randImpl(
      offset, ptr, len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        constexpr Type two = (Type)2.0;
        constexpr Type oneHalf = (Type)0.5;
        Type out;
        if (val <= oneHalf) {
          out = mu + scale * myLog(two * val);
        } else {
          out = mu - scale * myLog(two * (one - val));
        }
        return out;
      },
      NumThreads, nBlocks, type, stream);
  }

 private:
  /** generator type */
  GeneratorType type;
  /**
   * offset is also used to initialize curand state.
   * Limits period of Philox RNG from (4 * 2^128) to (Blocks * Threads * 2^64),
   * but is still a large period.
   */
  uint64_t offset;
  /** number of blocks to launch */
  int nBlocks;

  static const int NumThreads = 256;
};

};  // end namespace Random
};  // end namespace MLCommon
