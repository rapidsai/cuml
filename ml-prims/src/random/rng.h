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

template <typename OutType, typename GenType, typename Lambda>
__global__ void randKernel(uint64_t seed, uint64_t offset, OutType *ptr,
                           int len, Lambda randOp) {
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  detail::Generator<GenType> gen(seed, (uint64_t)tid, offset);
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned idx = tid; idx < len; idx += stride) {
    auto val = gen.next();
    ptr[idx] = (OutType)randOp(val, idx);
  }
}

// used for Box-Muller type transformations
template <typename OutType, typename GenType, typename Lambda2>
__global__ void rand2Kernel(uint64_t seed, uint64_t offset, OutType *ptr,
                            int len, Lambda2 rand2Op) {
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  detail::Generator<GenType> gen(seed, (uint64_t)tid, offset);
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned idx = tid; idx < len; idx += stride) {
    auto val1 = gen.next();
    auto val2 = gen.next();
    rand2Op(val1, val2, idx);
    if (idx < len)
      ptr[idx] = (OutType)val1;
    idx += stride;
    if (idx < len)
      ptr[idx] = (OutType)val2;
  }
}

template <bool IsNormal, typename Type>
uint64_t _setupSeeds(uint64_t &seed, uint64_t &offset, int len, int nThreads,
                     int nBlocks) {
  int itemsPerThread = ceildiv(len, nBlocks * nThreads);
  if (IsNormal && itemsPerThread % 2 == 1) {
    ++itemsPerThread;
  }
  // curand uses 2 32b uint's to generate one double
  uint64_t factor = sizeof(Type) / sizeof(float);
  if (factor == 0)
    ++factor;
  // Check if there are enough random numbers left in sequence
  // If not, then generate new seed and start from zero offset
  auto newOffset = offset + itemsPerThread * factor;
  if (newOffset < offset) {
    offset = 0;
    seed = _nextSeed();
    newOffset = itemsPerThread * factor;
  }
  return newOffset;
}

template <typename OutType, typename MathType = OutType, typename Lambda>
void randImpl(uint64_t &offset, OutType *ptr, int len, Lambda randOp,
              int nThreads, int nBlocks, GeneratorType type,
              cudaStream_t stream = 0) {
  if (len <= 0)
    return;
  uint64_t seed;
  auto newOffset =
    _setupSeeds<false, MathType>(seed, offset, len, nThreads, nBlocks);
  switch (type) {
    case GenPhilox:
      randKernel<OutType, detail::PhiloxGenerator<MathType>,
                 Lambda><<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr,
                                                           len, randOp);
      break;
    case GenTaps:
      randKernel<OutType, detail::TapsGenerator<MathType>,
                 Lambda><<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr,
                                                           len, randOp);
      break;
    case GenKiss99:
      randKernel<OutType, detail::Kiss99Generator<MathType>,
                 Lambda><<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr,
                                                           len, randOp);
      break;
    default:
      ASSERT(false, "randImpl: Incorrect generator type! %d", type);
  };
  CUDA_CHECK(cudaPeekAtLastError());
  offset = newOffset;
}

template <typename OutType, typename MathType = OutType, typename Lambda2>
void rand2Impl(uint64_t &offset, OutType *ptr, int len, Lambda2 rand2Op,
               int nThreads, int nBlocks, GeneratorType type,
               cudaStream_t stream = 0) {
  if (len <= 0)
    return;
  uint64_t seed;
  auto newOffset =
    _setupSeeds<true, MathType>(seed, offset, len, nThreads, nBlocks);
  switch (type) {
    case GenPhilox:
      rand2Kernel<OutType, detail::PhiloxGenerator<MathType>,
                  Lambda2><<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr,
                                                             len, rand2Op);
      break;
    case GenTaps:
      rand2Kernel<OutType, detail::TapsGenerator<MathType>,
                  Lambda2><<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr,
                                                             len, rand2Op);
      break;
    case GenKiss99:
      rand2Kernel<OutType, detail::Kiss99Generator<MathType>,
                  Lambda2><<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr,
                                                             len, rand2Op);
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


/**
 * @brief Random number generator
 * @tparam Type the data-type in which to return the random numbers
 */
template <typename Type>
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
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param start start of the range
   * @param end end of the range
   * @param stream stream where to launch the kernel
   */
  void uniform(Type *ptr, int len, Type start, Type end,
               cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) {
               return (end - start) * val + start;
             },
             NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate normal distributed numbers
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   */
  void normal(Type *ptr, int len, Type mu, Type sigma,
              cudaStream_t stream = 0) {
    rand2Impl(offset, ptr, len,
              [=] __device__(Type & val1, Type & val2, unsigned idx) {
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
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param val value to be filled
   * @param stream stream where to launch the kernel
   */
  void fill(Type *ptr, int len, Type val, cudaStream_t stream = 0) {
    constFillKernel<Type><<<nBlocks, NumThreads, 0, stream>>>(ptr, len, val);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  /**
   * @brief Generate bernoulli distributed boolean array
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param prob coin-toss probability for heads
   * @param stream stream where to launch the kernel
   */
  void bernoulli(bool *ptr, int len, Type prob, cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) { return val > prob; },
             NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate gumbel distributed random numbers
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param beta scale value
   * @param stream stream where to launch the kernel
   * @note https://en.wikipedia.org/wiki/Gumbel_distribution
   */
  void gumbel(Type *ptr, int len, Type mu, Type beta, cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) {
               return mu - beta * myLog(-myLog(val));
             },
             NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate lognormal distributed numbers
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   */
  void lognormal(Type *ptr, int len, Type mu, Type sigma,
                 cudaStream_t stream = 0) {
    rand2Impl(offset, ptr, len,
              [=] __device__(Type & val1, Type & val2, unsigned idx) {
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
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param scale scale value
   * @param stream stream where to launch the kernel
   */
  void logistic(Type *ptr, int len, Type mu, Type scale,
                cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) {
               constexpr Type one = (Type)1.0;
               return mu - scale * myLog(one / val - one);
             },
             NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate exponentially distributed random numbers
   * @param ptr output array
   * @param len number of elements in the output array
   * @param lambda the lambda
   * @param stream stream where to launch the kernel
   */
  void exponential(Type *ptr, int len, Type lambda, cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) {
               constexpr Type one = (Type)1.0;
               return -myLog(one - val) / lambda;
             },
             NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate rayleigh distributed random numbers
   * @param ptr output array
   * @param len number of elements in the output array
   * @param sigma the sigma
   * @param stream stream where to launch the kernel
   */
  void rayleigh(Type *ptr, int len, Type sigma, cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) {
               constexpr Type one = (Type)1.0;
               constexpr Type two = (Type)2.0;
               return mySqrt(-two * myLog(one - val)) * sigma;
             },
             NumThreads, nBlocks, type, stream);
  }

  /**
   * @brief Generate laplace distributed random numbers
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu the mean
   * @param scale the scale
   * @param stream stream where to launch the kernel
   */
  void laplace(Type *ptr, int len, Type mu, Type scale,
               cudaStream_t stream = 0) {
    randImpl(offset, ptr, len,
             [=] __device__(Type val, unsigned idx) {
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

}; // end namespace Random
}; // end namespace MLCommon
