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
#include <cub/cub.cuh>
#include <limits.h>
#include <linalg/unary_op.h>
#include <linalg/ternary_op.h>

#include "smo_sets.h"
#include "workingset.h"

#include <iostream>
/*
#include <linalg/gemv.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/norm.h>
#include <linalg/eltwise.h>

#include <linalg/cublas_wrappers.h>

#include <linalg/map_then_reduce.h>
*/


namespace ML {
namespace SVM {

using namespace MLCommon;


/**
* Implements SMO algorithm based on ThunderSVM and OHD-SVM. 
*/
template<typename math_t>
class SmoSolver {
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

  WorkingSet<math_t> *ws;
  //KernelCache cache;
  
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
  /*
  void BlockSolve() {
    int nIter = 0; 
    typedef cub::BlockReduce<math_t, 128> BlockReduce;

    while (nIter < maxIter) {
    // mask F values outside of X_upper  
      ternaryOp(tmp, f, alpha, y, int n_rows, 
         [C]__device__(math_t f, math_t a, math_t y){ return in_upper(a, y, C) ? f : std::numeric_limits<math_t>::max(); }
            );
     __syncthreads();//?
      int u = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMin());
      

//  	    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, tmp, &u, 1);

      //gpusvm simply selects based on f value
      ternaryOp(tmp, f, alpha, y, int n_rows,
                        [C]__device__(math_t f, math_t a, math_t y){ return in_lower(a, y, C) ? f : std::numeric_limits<math_t>::min(); }
                      );
      int l = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());
      
      //// ThunderSVM : l_formula needs to be implemented (f_u - f_i)^2/eta_i
      //ternaryOp(tmp, f, alpha, y, n_rows, 
      //            [C, u, f]__device__(math_t fval, math_t a, math_t y) { 
      //                return in_lower(a, y, c) && f[u] < fval ? l_formula(fval, f[u], eta) : std::numeric_limits<math_t>::max(); }
      //          );
      
      
      math_t alpha_l_new = alpha[l] + y[l] * (f[u] - f[l]) / eta(u,l);
      if (alpha_l_new < 0) alpha_l_new = 0;
      else if (alpha_l_new > c) alpha_l_new = C;
      Dl = (alpha_l_new - alpha[l]) * y[l];
      
      alpha_u_new = alpha[u] - y[l]*y[u]*alpha_l_diff;
      if (alpha_u_new < 0) alpha_u_new = 0;
      else if (alpha_u_new > c) alpha_u_new = C;
      Du = (alpha_u_new - alpha[u]) * y[u];
      
      // update coefficients
      ternaryOp(f, f, Ku, Kl, n_rows, 
                  [C, u, Dalpha_l, Dalpha_u]__device__(math_t f, math_t Ku, math_t Kl) { return + Du * Ku + Dl* Kl}
                );
      math_t fmax;
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, f, &fmax, 1); 
      if (f[l] > f[u] + 2*tol) { // error, f[u] should not be evaluated on the host
          break; 
    }
  }
  */
  ///
  /// Init the values of alpha, f, and helper buffers. 
  void Initialize(math_t* y) {
    // we initialize 
    // alpha_i = 0 and 
    // f_i = -y_i
    //CUDA_CHECK(cudaMemset((*void)alpha, 0, n_rows * sizeof(math_t)));
   // unaryOp(*f, y, int n_rows, []__device__(math_t f, math_t y){ return -y; });

  }

  
  void Solve(math_t* x, int n_rows, int n_cols, math_t* y, math_t **nz_alpha, int **idx) {
    int n_iter = 0;
    
    ws = new WorkingSet<math_t>(n_rows);
    int n_ws = ws->GetSize();
    AllocateBuffers();    
    
    Initialize(y);
    
    //cache = new KernelCache(x, n_rows, n_cols, n_ws);
    
    while (n_iter < 1) { // TODO: add proper stopping condition
      ws->Select(f, alpha, y, C);
      //math_t * cacheTile = cache->initTile(ws->idx); 
      CUDA_CHECK(cudaMemset(delta_alpha, 0, n_ws * sizeof(math_t)));
        
      //BlockSolve<1,ws,CALC_SHMEM_SIZE>(n_ws, f, y, alpha, delta_alpha, ws_idx, return_buff);

        
      // updateF();
        
      n_iter++;
    }    
    
    //delete cache;
    delete ws;
    FreeBuffers(); 
  }
};

}; // end namespace SVM 
}; // end namespace ML

