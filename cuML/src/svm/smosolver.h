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

#include <cuda_utils.h>

#include "smo_sets.h"
#include "workingset.h"
#include "kernelcache.h"
#include "smoblocksolve.h"
#include "linalg/gemv.h"

namespace ML {
namespace SVM {

using namespace MLCommon;


/**
* Implements SMO algorithm based on ThunderSVM and OHD-SVM. 
*/
template<typename math_t>
class SmoSolver {
private:
  int n_rows = 0;
  int n_ws = 0;
  // Buffers for the domain [n_rows]
  math_t *alpha = nullptr;       //< dual coordinates
  math_t *f = nullptr;           //< optimality indicator vector
  
  // Buffers for the working set [n_ws]
  math_t *delta_alpha = nullptr; // change for the working set
  
  // return some parameters from the kernel;
  math_t *return_buff = nullptr;
  math_t host_return_buff[2];  // used to return iteration numbef and convergence information from the kernel
  
  math_t C;
  math_t tol;
public:
  SmoSolver(math_t C = 1, math_t tol = 0.001) 
    : n_rows(n_rows), C(C), tol(tol)
  {
  }
  
  ~SmoSolver() {
      FreeBuffers();
  }
  

  // this needs to know n_ws, therefore it can be only called during the solve step
  void AllocateBuffers() {
    FreeBuffers();
    this->n_rows=n_rows;
    allocate(alpha, n_rows); 
    allocate(f, n_rows);  
    allocate(delta_alpha, n_ws);
    allocate(return_buff, 2);
  }    
  
  void FreeBuffers() {
    if(alpha) CUDA_CHECK(cudaFree(alpha));
    if(f) CUDA_CHECK(cudaFree(f));
    if(delta_alpha) CUDA_CHECK(cudaFree(delta_alpha));
    if(return_buff) CUDA_CHECK(cudaFree(return_buff));
    alpha = nullptr;
    f = nullptr;
    delta_alpha = nullptr;
    return_buff = nullptr;
  }
    
  ///
  /// Init the values of alpha, f, and helper buffers. 
  void Initialize(math_t* y) {
    // we initialize 
    // alpha_i = 0 and 
    // f_i = -y_i
    CUDA_CHECK(cudaMemset(alpha, 0, n_rows * sizeof(math_t)));
    LinAlg::unaryOp(f, y, n_rows, []__device__(math_t y){ return -y; });

  }

  
  void Solve(math_t* x, int n_rows, int n_cols, math_t* y, math_t **nz_alpha, int **idx,
      cublasHandle_t cublas_handle) {
    int n_iter = 0;
    
    WorkingSet<math_t> ws(n_rows);
    int n_ws = ws.GetSize();
    AllocateBuffers();    
    Initialize(y);
    
    KernelCache<math_t> cache(x, n_rows, n_cols, n_ws, cublas_handle);
    
    while (n_iter < 1) { // TODO: add proper stopping condition
      CUDA_CHECK(cudaMemset(delta_alpha, 0, n_ws * sizeof(math_t)));
      ws.Select(f, alpha, y, C);
      math_t * cacheTile = cache.GetTile(ws.idx); 
      
      SmoBlockSolve<math_t, 1024><<<1, n_ws>>>(y, n_ws, alpha, delta_alpha, f, cacheTile,
                                  ws.idx, C, tol, return_buff);
      updateHost(host_return_buff, return_buff, 2);
        
      UpdateF(f, n_rows, delta_alpha, n_ws, cacheTile, cublas_handle);
      // check stopping condition
      math_t diff = host_return_buff[0];
      n_iter++;
    }    
    
    FreeBuffers(); 
  }
  
  void UpdateF(math_t *f, const int n_rows, const math_t *delta_alpha, int n_ws, const math_t *cacheTile, cublasHandle_t cublas_handle) {
    // check sign here too.
    LinAlg::gemv(cacheTile, n_ws, n_rows, delta_alpha, 1, f, 1, true, math_t(-1.0), math_t(1.0), cublas_handle);
  }
};

}; // end namespace SVM 
}; // end namespace ML

