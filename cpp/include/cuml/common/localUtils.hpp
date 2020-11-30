/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "utils.hpp"

namespace MLCommon {

__device__ inline void print_tid() {
  printf("threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
}

template<typename T>
void printHostArray(const T* hostArray, int length, const char* name="") {
  
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Location of array = %p ", hostArray);
  if(name != "") { 
    printf("%s = {", name);
  }
  else {
    printf("{");
  }
  for(int i = 0; i < length; i++)  {
      std::cout << hostArray[i] << ", ";
      // printf("%f, ", hostArray[i]);
  }
  printf("}\n");
  fflush(stdout);
}

template<typename T>
void printDeviceArray(const T* deviceArray, int length, const char* name="") {
  
  T *hostArray = nullptr;
  hostArray =  (T*) malloc(sizeof(T)*length);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hostArray, deviceArray,
    sizeof(T)*length, cudaMemcpyDeviceToHost));
  // printf("Location of array = %p ", deviceArray);
  // if(name != "") { 
  //   printf("%s = {", name);
  // }
  // else {
  //   printf("{");
  // }
  // for(int i = 0; i < length; i++)  {
  //     std::cout << hostArray[i] << ", ";
  //     // printf("%f, ", hostArray[i]);
  // }
  // printf("}\n");
  printHostArray(hostArray, length, name);
  free(hostArray);

}

};  // namespace MLCommon

