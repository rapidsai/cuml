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
#include "cub/cub.cuh"
#include <utils.h>

namespace DecisionTree {
  
  void histogram(float *d_samples, int *d_histogram, int num_levels,float lower_level,float upper_level,int num_samples)
  {
    void*    d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    
    //first call to compute temp storage, no kernel launched
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
						  d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples));  
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
						  d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples));
    CUDA_CHECK(cudaFree(d_temp_storage));
    
  }
  
}// end namespace DecisionTree
