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

#include "cuda_utils.h"
#include <cstdlib>
#include <cstdio>


namespace MLCommon {
namespace Random {

//Courtesy: VinayD
#define TAPS 0x8000100040002000ULL
template <typename Type>
DI Type randVal(unsigned long long& state) {
    Type res;
    for (int i=0;i<128;i++)
        state = (state >> 1) ^ (-(state & 1ULL) & TAPS);
    res = static_cast<Type>(state);
    res /= static_cast<Type>(1.8446744073709551614e19);
    return res;
}

template <typename Type, int ItemsPerThread>
__global__ void uniformKernel(unsigned long long seed, Type* ptr, int len,
                              Type start, Type end) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                ptr[idx] = ((end - start) * t) + start;
                            });
}

template <typename Type, int ItemsPerThread, typename Lambda>
__global__ void normalKernel(unsigned long long seed, Type* ptr, int len,
                             Type mu, Type sigma, Lambda op) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    constexpr Type twoPi = Type(2.0) * Type(3.141592654);
    constexpr Type minus2 = -Type(2.0);
    #pragma unroll
    for(int itr=0;itr<ItemsPerThread;itr+=2) {
        Type u1 = randVal<Type>(state);
        Type u2 = randVal<Type>(state);
        Type R = mySqrt(minus2 * myLog(u1));
        Type theta = twoPi * u2;
        Type s, c;
        mySinCos(theta, s, c);
        Type val1 = R * c * sigma + mu;
        Type val2 = R * s * sigma + mu;
        op(val1, val2);
        if(idx < len)
            ptr[idx] = val1;
        idx += numThreads;
        if(idx < len)
            ptr[idx] = val2;
        idx += numThreads;
    }
}

template <typename Type, int ItemsPerThread>
__global__ void constFillKernel(Type* ptr, int len, Type val) {
    forEach<ItemsPerThread>(len,
                            [=] __device__ (int idx, int itr) {
                                ptr[idx] = val;
                            });
}

template <typename Type, int ItemsPerThread>
__global__ void bernoulliKernel(unsigned long long seed, bool* ptr, int len,
                                Type prob) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                ptr[idx] = t > Type(prob);
                            });
}

template <typename Type, int ItemsPerThread>
__global__ void gumbelKernel(unsigned long long seed, Type* ptr, int len,
                             Type mu, Type beta) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                ptr[idx] = mu - beta * myLog(-myLog(t));
                            });
}

template <typename Type, int ItemsPerThread>
__global__ void logisticKernel(unsigned long long seed, Type* ptr, int len,
                               Type mu, Type scale) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    const Type one = Type(1.0);
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                ptr[idx] = mu - scale * myLog(one/t - one);
                            });
}

template <typename Type, int ItemsPerThread>
__global__ void expKernel(unsigned long long seed, Type* ptr, int len,
                          Type lambda) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    const Type one = Type(1.0);
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                ptr[idx] = -myLog(one - t) / lambda;
                            });
}

template <typename Type, int ItemsPerThread>
__global__ void rayleighKernel(unsigned long long seed, Type* ptr, int len,
                               Type sigma) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    const Type one = Type(1.0);
    const Type two = Type(2.0);
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                ptr[idx] = mySqrt(-two * myLog(one - t)) * sigma;
                            });
}

template <typename Type, int ItemsPerThread>
__global__ void laplaceKernel(unsigned long long seed, Type* ptr, int len,
                              Type mu, Type scale) {
    unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned long long state = seed + tid + 1;
    const Type one = Type(1.0);
    const Type two = Type(2.0);
    const Type oneHalf = Type(0.5);
    forEach<ItemsPerThread>(len,
                            [&] __device__(int idx, int itr) {
                                Type t = randVal<Type>(state);
                                Type out;
                                if(t <= oneHalf) {
                                    out = mu + scale * myLog(two * t);
                                } else {
                                    out = mu - scale * myLog(two * (one - t));
                                }
                                ptr[idx] = out;
                            });
}


/**
 * @brief Random number generator
 * @tparam Type the data-type in which to return the random numbers
 * @note Why not curand? It's state maintenance overhead is too high
 *  Moreover, with a primitive-prime-polynomial over GF(2^64), we can
 *  very easily get a RNG with cycle of 2^64!
 */
template <typename Type>
class Rng {
public:
    /** ctor */
    Rng(unsigned long long _s) {
        srand(_s);
    }

    /**
     * @brief Generate uniformly distributed numbers in the given range
     * @param ptr the output array
     * @param len the number of elements in the output
     * @param start start of the range
     * @param end end of the range
     */
    void uniform(Type* ptr, int len, Type start, Type end) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        uniformKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, start, end);
    }

    /**
     * @brief Generate normal distributed numbers
     * @param ptr the output array
     * @param len the number of elements in the output
     * @param mu mean of the distribution
     * @param sigma std-dev of the distribution
     */
    void normal(Type* ptr, int len, Type mu, Type sigma) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        normalKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, mu, sigma,
             [] __device__(Type& val1, Type& val2) {});
    }

    /**
     * @brief Fill an array with the given value
     * @param ptr the output array
     * @param len the number of elements in the output
     * @param val value to be filled
     */
    void fill(Type* ptr, int len, Type val) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        constFillKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (ptr, len, val);
    }

    /**
     * @brief Generate bernoulli distributed boolean array
     * @param ptr the output array
     * @param len the number of elements in the output
     * @param prob coin-toss probability for heads
     */
    void bernoulli(bool* ptr, int len, Type prob) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        bernoulliKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, prob);
    }

    /**
     * @brief Generate gumbel distributed random numbers
     * @param ptr output array
     * @param len number of elements in the output array
     * @param mu mean value
     * @param beta scale value
     * @note https://en.wikipedia.org/wiki/Gumbel_distribution
     */
    void gumbel(Type* ptr, int len, Type mu, Type beta) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        gumbelKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, mu, beta);
    }

    /**
     * @brief Generate lognormal distributed numbers
     * @param ptr the output array
     * @param len the number of elements in the output
     * @param mu mean of the distribution
     * @param sigma std-dev of the distribution
     */
    void lognormal(Type* ptr, int len, Type mu, Type sigma) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        normalKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, mu, sigma,
             [] __device__(Type& val1, Type& val2) {
                 val1 = myExp(val1);
                 val2 = myExp(val2);
             });
    }

    /**
     * @brief Generate logistic distributed random numbers
     * @param ptr output array
     * @param len number of elements in the output array
     * @param mu mean value
     * @param scale scale value
     */
    void logistic(Type* ptr, int len, Type mu, Type scale) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        logisticKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, mu, scale);
    }

    /**
     * @brief Generate exponentially distributed random numbers
     * @param ptr output array
     * @param len number of elements in the output array
     * @param lambda the lambda
     */
    void exponential(Type* ptr, int len, Type lambda) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        expKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, lambda);
    }

    /**
     * @brief Generate rayleigh distributed random numbers
     * @param ptr output array
     * @param len number of elements in the output array
     * @param sigma the sigma
     */
    void rayleigh(Type* ptr, int len, Type sigma) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        rayleighKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, sigma);
    }

    /**
     * @brief Generate laplace distributed random numbers
     * @param ptr output array
     * @param len number of elements in the output array
     * @param mu the mean
     * @param scale the scale
     */
    void laplace(Type* ptr, int len, Type mu, Type scale) {
        int blkDim = ceildiv(len, ItemsPerBlk);
        laplaceKernel<Type, ItemsPerThread><<<blkDim, NumThreads>>>
            (nextSeed(), ptr, len, mu, scale);
    }

private:
    unsigned long long nextSeed() const {
        unsigned long long t1 = (unsigned long long) (rand() + rand());
        unsigned long long t2 = (unsigned long long) (rand() + rand());
        return ((t2 << 32) | t1);
    }

    static const int ItemsPerThread = 4;
    static const int NumThreads = 256;
    static const int ItemsPerBlk = NumThreads * ItemsPerThread;
};

}; // end namespace Random
}; // end namespace MLCommon
