/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <raft/linalg/norm.cuh>

#include <cuda_runtime.h>
#include <cuml/cuml.hpp>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <sys/time.h>
#include <raft/random/rng.cuh>
#include <raft/stats/sum.cuh>

#include <unistd.h>
#include <chrono>
#include <iostream>

/**
 * @brief Performs P + P.T.
 * @param[out] vector: The output vector you want to overwrite with randomness.
 * @param[in] minimum: The minimum value in the output vector you want.
 * @param[in] maximum: The maximum value in the output vector you want.
 * @param[in] size: The size of the output vector.
 * @param[in] stream: The GPU stream.
 * @param[in] seed: If seed == -1, then the output is pure randomness. If >= 0, then you can reproduce TSNE.
 */
void random_vector(float *vector, const float minimum, const float maximum,
                   const int size, cudaStream_t stream, long long seed = -1) {
  if (seed <= 0) {
    // Get random seed based on time of day
    struct timeval tp;
    gettimeofday(&tp, NULL);
    seed = tp.tv_sec * 1000 + tp.tv_usec;
  }
  raft::random::Rng random(seed);
  random.uniform<float>(vector, size, minimum, maximum, stream);
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
  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {                   \
    gettimeofday(&timecheck, NULL);                                         \
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
  }

#define END_TIMER(add_onto)                                               \
  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {                 \
    gettimeofday(&timecheck, NULL);                                       \
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
    add_onto += (end - start);                                            \
  }

#define PRINT_TIMES                                                           \
  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {                     \
    double total =                                                            \
      (SymmetrizeTime + DistancesTime + NormalizeTime + PerplexityTime +      \
       BoundingBoxKernel_time + ClearKernel1_time + TreeBuildingKernel_time + \
       ClearKernel2_time + SummarizationKernel_time + SortKernel_time +       \
       RepulsionTime + Reduction_time + attractive_time +                     \
       IntegrationKernel_time) /                                              \
      100.0;                                                                  \
    CUML_LOG_DEBUG(                                                           \
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
      "TOTAL TIME = %.lf",                                                    \
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
