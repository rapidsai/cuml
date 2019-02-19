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

/** Determines weather a training instance is in the upper set */
template<typename math_t> 
DI bool in_upper(math_t a, math_t y, math_t C) {
  // return (0 < a && a < C) || ((y - 1) < eps && a < eps) || ((y + 1) < eps && (a - C) < eps);
  // since a is always clipped to lie in the [0 C] region, therefore this is equivalent with
  return (y < 0 && a > 0) || (y > 0 && a < C);
}

/** determines weather a training instance is in the lower set */
template<typename math_t> 
DI bool in_lower(math_t a, math_t y, math_t C) {
  // return (0 < a && a < C) || ((y - 1) < eps && a < eps) || ((y + 1) < eps && (a - C) < eps);
  // since a is always clipped to lie in the [0 C] region, therefore this is equivalent with
  return (y < 0 && a < C) || (y > 0 && a > 0);
}

template <typename math_t>
__global__ void init_smo_buffers(int n_rows, int* f_idx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n_rows) {
    f_idx[idx] = idx;
  }
}

/**
* Implements SMO algorithm based on ThunderSVM and OHD-SVM. 
*/
template<typename math_t>
class SmoSolver {
  int n_rows;
  int n_ws;
  // Buffers for the domain [n_rows]
  math_t *alpha;       //< dual coordinates
  math_t *f;           //< optimality indicator vector
  int *f_idx;          //< Arrays used for sorting f
  int *f_idx_sorted;   //<
  
  // Buffers for the working set [n_ws]
  math_t *delta_alpha; // change for the working set
  int *ws_idx; // indices for elements in the working set
  math_t return_buff[2];  // used to return iteration numbef and convergence information from the kernel
  void *cub_temp_storage = NULL; // used by cub for reduction
  int cub_temp_storage_bytes;
  
public:
  SmoSolver(int n_rows, int ws_size)
    : n_rows(n_rows), n_ws(n_ws) {
      AllocateBuffers();
  }
  
  ~SmoSolver() {
      FreeBuffers();
  }
  

  void AllocateBuffers() {
    allocate(alpha, n_rows); 
    allocate(f, n_rows);  
    allocate(f_idx, n_rows);     
    allocate(f_idx_sorted, n_rows);
    //allocate(tmp, n_rows);   
    allocate(delta_alpha, n_ws);
    allocate(ws_idx, n_ws);

    // Determine temporary device storage requirements for cub
    cub_temp_storage = NULL;
    cub_temp_storage_bytes = 0;
    //cub::DeviceRadixSort::SortPairs(cub_temp_storage, cub_temp_storage_bytes, f_idx, f_idx, 
    //                                f, f, n_rows);
    allocate(cub_temp_storage, cub_temp_storage_bytes);
  }    
  
  void FreeBuffers() {
    CUDA_CHECK(cudaFree(alpha));
    CUDA_CHECK(cudaFree(f));
    CUDA_CHECK(cudaFree(f_idx));
    CUDA_CHECK(cudaFree(f_idx_sorted));
    //CUDA_CHECK(cudaFree(tmp));
    CUDA_CHECK(cudaFree(delta_alpha));
    CUDA_CHECK(cudaFree(cub_temp_storage));
    CUDA_CHECK(cudaFree(ws_idx));
  }
  /*
  void SelectWorkingSet() {
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(cub_temp_storage, cub_temp_storage_bytes,
        f_idx, f_idx_sorted, f, f_sorted, n_rows);
      
    int n_selected;
    // Run selection
    cub::DeviceSelect::If(cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted, idx_tmp, n_selected, n_rows, 
        [alpha, y, C](idx) { return in_upper(alpha[idx], y[idx], C); }
    );
      
    CUDA_CHECK(cudaMemcpy(ws_idx, idx_tmp, n_ws/2 * sizeof(int), cudaMemcpyDeviceToDevice));
      
    cub::DeviceSelect::If(cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted, idx_tmp, n_selected, n_rows, 
        [alpha, y, C](idx) { return in_lower(alpha[idx], y[idx], C); }
    );
      
    CUDA_CHECK(cudaMemcpy(ws_idx + n_ws/2, idx_tmp+n_selected-n_ws/2, n_ws/2 * sizeof(int), cudaMemcpyDeviceToDevice));
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

  ///
  /// Init the values of alpha, f, and helper buffers. 
  void Initialize(math_t* y) {
    // we initialize 
    // alpha_i = 0 and 
    // f_i = -y_i
    CUDA_CHECK(cudaMemset(*alpha, 0, n_rows * sizeof(mem_t)));
    unaryOp(*f, y, int n_rows, []__device__(math_t f, math_t y){ return -y; });

    int TPB = 256;
    init_smo_buffers<<<ceildiv(n_rows, TPB), TPB>>>(n_rows, *f_idx);

  }
  */
  
  void Solve(math_t* x, math_t* y, math_t **nz_alpha, int **idx) {
    int n_iter = 0;
    int * idx_tmp;
    
    //Initialize(y);
        
    while (n_iter < 1) { // TODO: add proper stopping condition
   /*     SelectWorkingSet(n_rows, n_ws, y, f, f_idx, f_idx_sorted, alpha, 
                 cub_temp_storage,  cub_temp_storage_bytes, ws_idx );
        
        //initCacheTile(); 

        CUDA_CHECK(cudaMemset(delta_alpha, 0, n_ws * sizeof(mem_t)));
     */   
        //BlockSolve<1,ws,CALC_SHMEM_SIZE>(n_ws, f, y, alpha, delta_alpha, ws_idx, return_buff);

        
       // updateF;
        
       n_iter++;
    }    
  }
};

}; // end namespace SVM 
}; // end namespace ML

