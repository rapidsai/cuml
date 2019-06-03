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

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "utils.h"

#pragma once

namespace ML {

template<typename T = size_t>
bool verify_size(T size, int device) {
      size_t free, total;
      cudaMemGetInfo(&free, &total);

      if(size > free) {
          std::cout << "Not enough free memory on device "
                  << device
                  << ". needed="
                  << size
                  << ", free=" << free << std::endl;
          return false;
      }

      return true;
};



template<typename T>
void ASSERT_MEM(T *ptr, std::string name) {
      cudaPointerAttributes s_att;
      cudaError_t s_err = cudaPointerGetAttributes(&s_att, ptr);

      if(s_err != 0 || s_att.device == -1)
          std::cout << "Invalid device pointer encountered in " << name <<
                    ". device=" << s_att.device << ", err=" << s_err << std::endl;
};


int get_device(const void *ptr) {
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att, ptr);
    return att.device;
}

cudaMemoryType memory_type(const void *p) {
        cudaPointerAttributes att;
        cudaError_t err = cudaPointerGetAttributes(&att, p);
        ASSERT(err == cudaSuccess ||
               err == cudaErrorInvalidValue, "%s", cudaGetErrorString(err));

        if (err == cudaErrorInvalidValue) {
            // Make sure the current thread error status has been reset
            err = cudaGetLastError();
            ASSERT(err == cudaErrorInvalidValue, "%s", cudaGetErrorString(err));
        }
    #if CUDA_VERSION >= 10000
        return att.type;
    #else
        return att.memoryType;
    #endif
  }
}
