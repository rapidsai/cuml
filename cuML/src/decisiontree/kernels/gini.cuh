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
#include <utils.h>
#include "cub/cub.cuh"
#include <thrust/sort.h>
#include "../memory.cuh"

float gini(int *labels_in,const int nrows,const TemporaryMemory* tempmem,const cudaStream_t stream = 0)
{
  float gval = 1.0;
  int *labels = tempmem->ginilabels;
  
  CUDA_CHECK(cudaMemcpyAsync(labels,labels_in,nrows*sizeof(int),cudaMemcpyDeviceToDevice,stream));
  
  thrust::sort(thrust::cuda::par.on(stream),labels,labels + nrows);
  
  // Declare, allocate, and initialize device-accessible pointers for input and output
  int *d_unique_out = tempmem->d_unique_out;      
  int *d_counts_out = tempmem->d_counts_out;      
  int *d_num_runs_out = tempmem->d_num_runs_out;    
  
  // Determine temporary device storage requirements
  void     *d_temp_storage = tempmem->d_gini_temp_storage;
  size_t temp_storage_bytes = tempmem->gini_temp_storage_bytes;
  
  // Run encoding
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, labels, d_unique_out, d_counts_out, d_num_runs_out, nrows, stream));

  int num_unique;
  CUDA_CHECK(cudaMemcpyAsync(&num_unique,d_num_runs_out,sizeof(int),cudaMemcpyDeviceToHost,stream));

  int *h_counts_out = (int*)malloc(num_unique*sizeof(int));

  CUDA_CHECK(cudaMemcpyAsync(h_counts_out,d_counts_out,num_unique*sizeof(int),cudaMemcpyDeviceToHost,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  
  for(int i=0;i<num_unique;i++)
    {
      float prob = ((float)h_counts_out[i]) / nrows;
      gval -= prob*prob; 
    }

  
  free(h_counts_out);
  
  
  return gval;
}
  
