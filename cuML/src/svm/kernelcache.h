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

#include "ml_utils.h"
#include <cuda_utils.h>
#include <linalg/gemm.h>

namespace ML {
namespace SVM {

using namespace MLCommon;

/**
* Buffer to store a kernel tile.
*/ 
template<typename math_t>
class KernelCache {
  math_t *x;
  int n_rows;
  int n_cols;
  int n_ws;
  math_t *tile = nullptr;
  cublasHandle_t cublas_handle;
  
public:
  KernelCache(math_t *x, int n_rows, int n_cols, int n_ws, cublasHandle_t cublas_handle) 
    : x(x), n_rows(n_rows), n_cols(n_cols), n_ws(n_ws), cublas_handle(cublas_handle)
  {
    allocate(tile, n_rows*n_ws);
  };
  ~KernelCache() {
    CUDA_CHECK(cudaFree(tile));
  };
  math_t* GetTile(int *ws_idx) {
    // implementing only linear kernel so far
    // we need to gather ws_idx rows
    // lets assume that it is collected into
    math_t *x_ws = x; // //CollectRows(ws_idx);
    //calculate kernel function values for indices in ws_idx
    LinAlg::gemm(x, n_ws, n_cols, x, tile, n_ws, n_ws, CUBLAS_OP_N,
          CUBLAS_OP_T, math_t(1.0), math_t(0.0), cublas_handle) ;
    return tile;
  }
};

}; // end namespace SVM 
}; // end namespace ML
