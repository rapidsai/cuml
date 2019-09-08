/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#include "common/cumlHandle.hpp"
#include "linalg/norm.h"

#include <cuda_runtime.h>
#include <cuda_utils.h>
#include "cuML.hpp"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <random/rng.h>
#include <stats/sum.h>
#include <sys/time.h>

#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

/**
 * @brief Random Vector intialization.
 * @output param vector: The output vector you want to overwrite with randomness.
 * @input param minimum: The minimum value in the output vector you want.
 * @input param maximum: The maximum value in the output vector you want.
 * @input param size: The size of the output vector.
 * @input param stream: The GPU stream.
 * @input param seed: If seed == -1, then the output is pure randomness. If >= 0, then you can reproduce TSNE.
 * @input param normal: If true, then will provide random normal numbers
 */
template <typename T> void
random_vector(T *vector,
              const T minimum, // mean for normal == true
              const T maximum, // std for normal == true
              const int size,
              cudaStream_t stream,
              long long seed = -1,
              const bool normal = false)
{
  if (seed <= 0) {
    // Get random seed based on time of day
    struct timeval tp;
    gettimeofday(&tp, NULL);
    seed = tp.tv_sec * 1000 + tp.tv_usec;
  }

  MLCommon::Random::Rng random(seed);
  if (not normal) {
    random.uniform<T>(vector, size, minimum, maximum, stream);
  }
  else {
    random.normal<T>(vector, size, minimum, maximum, stream);
  }

  CUDA_CHECK(cudaPeekAtLastError());
}



template <typename T>
__global__ static void
reverse_array(T *__restrict x,
              const int n)
{
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i > n / 2) return;

  const T left = x[i];
  x[i] = x[n-i-1]; // -1 as if n = 3, i = 0, X[n-i] = X[3-0]
  x[n-i-1] = left;
}


template <typename T>
__global__ static void
reverse_matrix(T *__restrict X, // F-Contiguous
               const int n,
               const int p)
{
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (j > p / 2) return;

  for (int i = 0; i < n; i++)
  {
    const T left = X[i + j*n];
    X[i + j*n] = X[i + (p-j-1)*n];
    X[i + (p-j-1)*n] = left;
  }
}


template <typename T>
void
reverse(T *__restrict X,
        const int n,
        const int p, // 0 means array.
        cudaStream_t stream)
{
  if (p == 0) {
    if (n == 1)
      return;

    reverse_array<<<MLCommon::ceildiv(n / 2, 1024), 1024, 0, stream>>>(X, n);
  }
  else {
    if (p == 1)
      return;

    reverse_matrix<<<MLCommon::ceildiv(p / 2, 1024), 1024, 0, stream>>>(X, n, p);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}


/*
X / array 
default is column wise divide ie:
X(n,p) / array(p)
*/
template <typename T>
__global__ static void
_matrix_multiply_by_array(T *__restrict X, // F-Contiguous
                          const int n,
                          const int p,
                          const T *__restrict array)
{
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (j >= p) return;

  const T mult = array[j];

  for (int i = 0; i < n; i++)
    X[i + j*n] *= mult;
}


template <typename T>
void
matrix_multiply_by_array(T *__restrict X, // F-Contiguous
                         const int n,
                         const int p,
                         const T *__restrict array,
                         cudaStream_t stream)
{
  _matrix_multiply_by_array<<<MLCommon::ceildiv(p, 1024), 1024, 0, stream>>>(X, n, p, array);
  CUDA_CHECK(cudaPeekAtLastError());
}





long start, end;
struct timeval timecheck;
double SymmetrizeTime = 0, DistancesTime = 0, NormalizeTime = 0,
       PerplexityTime = 0, BoundingBoxKernel_time = 0, ClearKernel1_time = 0,
       TreeBuildingKernel_time = 0, ClearKernel2_time = 0,
       SummarizationKernel_time = 0, SortKernel_time = 0, RepulsionTime = 0,
       Reduction_time = 0, attractive_time = 0, IntegrationKernel_time = 0;

// To silence warnings

#define START_TIMER                                                         \
  if (verbose) {                                                            \
    gettimeofday(&timecheck, NULL);                                         \
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
  }

#define END_TIMER(add_onto)                                               \
  if (verbose) {                                                          \
    gettimeofday(&timecheck, NULL);                                       \
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
    add_onto += (end - start);                                            \
  }

#define PRINT_TIMES                                                           \
  if (verbose) {                                                              \
    double total =                                                            \
      (SymmetrizeTime + DistancesTime + NormalizeTime + PerplexityTime +      \
       BoundingBoxKernel_time + ClearKernel1_time + TreeBuildingKernel_time + \
       ClearKernel2_time + SummarizationKernel_time + SortKernel_time +       \
       RepulsionTime + Reduction_time + attractive_time +                     \
       IntegrationKernel_time) /                                              \
      100.0;                                                                  \
    printf(                                                                   \
      "SymmetrizeTime = %.lf (%.lf)\n"                                        \
      "DistancesTime = %.lf (%.lf)\n"                                         \
      "NormalizeTime = %.lf (%.lf)\n"                                         \
      "PerplexityTime = %.lf (%.lf)\n"                                        \
      "BoundingBoxKernel_time = %.lf (%.lf)\n"                                \
      "ClearKernel1_time  = %.lf (%.lf)\n"                                    \
      "TreeBuildingKernel_time  = %.lf (%.lf)\n"                              \
      "ClearKernel2_time  = %.lf (%.lf)\n"                                    \
      "SummarizationKernel_time  = %.lf (%.lf)\n"                             \
      "SortKernel_time  = %.lf (%.lf)\n"                                      \
      "RepulsionTime  = %.lf (%.lf)\n"                                        \
      "Reduction_time  = %.lf (%.lf)\n"                                       \
      "attractive_time  = %.lf (%.lf)\n"                                      \
      "IntegrationKernel_time = %.lf (%.lf)\n"                                \
      "TOTAL TIME = %.lf\n\n",                                                \
      SymmetrizeTime, SymmetrizeTime / total, DistancesTime,                  \
      DistancesTime / total, NormalizeTime, NormalizeTime / total,            \
      PerplexityTime, PerplexityTime / total, BoundingBoxKernel_time,         \
      BoundingBoxKernel_time / total, ClearKernel1_time,                      \
      ClearKernel1_time / total, TreeBuildingKernel_time,                     \
      TreeBuildingKernel_time / total, ClearKernel2_time,                     \
      ClearKernel2_time / total, SummarizationKernel_time,                    \
      SummarizationKernel_time / total, SortKernel_time,                      \
      SortKernel_time / total, RepulsionTime, RepulsionTime / total,          \
      Reduction_time, Reduction_time / total, attractive_time,                \
      attractive_time / total, IntegrationKernel_time,                        \
      IntegrationKernel_time / total, total * 100.0);                         \
  }
