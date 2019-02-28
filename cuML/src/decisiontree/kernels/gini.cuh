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

float gini(int *labels,const int nrows)
{
  float gval = 1.0;
  thrust::sort(thrust::device,labels,labels + nrows);
  
  // Declare, allocate, and initialize device-accessible pointers for input and output
  int *d_unique_out;      
  int *d_counts_out;      
  int *d_num_runs_out;    
  
  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, labels, d_unique_out, d_counts_out, d_num_runs_out, nrows));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc((void**)(&d_unique_out),nrows*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)(&d_counts_out),nrows*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)(&d_num_runs_out),sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run encoding
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, labels, d_unique_out, d_counts_out, d_num_runs_out, nrows));

  int num_unique;
  CUDA_CHECK(cudaMemcpy(&num_unique,d_num_runs_out,sizeof(int),cudaMemcpyDeviceToHost));

  //int *h_unique_out = (int*)malloc(num_unique*sizeof(int));
  int *h_counts_out = (int*)malloc(num_unique*sizeof(int));
  //CUDA_CHECK(cudaMemcpy(h_unique_out,d_unique_out,num_unique*sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_counts_out,d_counts_out,num_unique*sizeof(int),cudaMemcpyDeviceToHost));
  
  for(int i=0;i<num_unique;i++)
    {
      float prob = ((float)h_counts_out[i]) / nrows;
      gval -= prob*prob; 
    }

  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_unique_out));
  CUDA_CHECK(cudaFree(d_counts_out));
  CUDA_CHECK(cudaFree(d_num_runs_out));
  free(h_counts_out);
  //free(h_unique_out);
    
  return gval;
}
  
