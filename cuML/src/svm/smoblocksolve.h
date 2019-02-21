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

//#include <utility>
#include "ml_utils.h"
#include <cuda_utils.h>
#include "selection/kselection.h"
#include "smo_sets.h"

namespace ML {
namespace SVM {

#define MAX_ITER 10000
  template<typename math_t, int WSIZE>
  __global__ void SmoBlockSolve(math_t *y_array, int n_ws, math_t* alpha, 
      math_t *delta_alpha, math_t *f_array, math_t *kernel, int *ws_idx, 
      math_t C, math_t eps, math_t *return_buff)
  {
    //typedef std::pair<math_t, int> Pair;
    typedef Selection::KVPair<math_t, int> Pair;
    typedef cub::BlockReduce<Pair, WSIZE> BlockReduce;
    typedef cub::BlockReduce<math_t, WSIZE> BlockReduceFloat;
    __shared__ union {
        typename BlockReduce::TempStorage pair; 
        typename BlockReduceFloat::TempStorage single;
    } temp_storage; 
    //__shared__ typename BlockReduce::TempStorage temp_storage;
    //__shared__ typename BlockReduceFloat::TempStorage temp_storage;
    
    //__shared__ math_t f_u;
    
    __shared__ math_t tmp1, tmp2;
    __shared__ math_t Kd[WSIZE]; // diagonal elements of the kernel matrix
    
    int tid = threadIdx.x; 
    int idx = ws_idx[tid];
    
    // store values in registers
    math_t y = y_array[idx];
    math_t f = f_array[idx];
    math_t a = alpha[idx];
    math_t a_save = a;
    math_t diff_end;
    
    Kd[tid] = kernel[tid*n_ws + tid]; //kernel[idx*n_ws + idx];
    
    for (int n_iter=0; n_iter < MAX_ITER; n_iter++) {
      // mask values outside of X_upper  
      math_t f_tmp = in_upper(a, y, C) ? f : INFINITY; 
      Pair pair{f_tmp, tid};
      Pair res = BlockReduce(temp_storage.pair).Reduce(pair, cub::Min());
      int u = res.key;
      math_t f_u = res.val;
      //if( tid==u) f_u = f;
      math_t Kui = kernel[u * n_ws + tid];
      // select f_max to check stopping condition
      f_tmp = in_lower(a, y, C) ? f : -INFINITY;
     __syncthreads();   // needed because I am reusing the shared memory buffer   
      math_t f_max = BlockReduceFloat(temp_storage.single).Reduce(f_tmp, cub::Max());
      
      // f_max-f_u is used to check stopping condition.
      math_t diff = f_max-f_u;
      if (n_iter==0) {
        if(tid==0) return_buff[0] = diff;
        // are the fmin/max functions overloaded for float/double?
        diff_end = max(eps, 0.1f*diff);
      }
      
      if (diff < diff_end || n_iter == MAX_ITER-1) {
        // save registers to global memory before exit
        alpha[idx]  = a;
        delta_alpha[tid] = - (a - a_save) * y;
        return_buff[1] = n_iter;
        break;
      }
       
      if (f_u < f && in_lower(a, y, C)) {
        f_tmp = (f_u - f) * (f_u - f) / (Kd[tid] + Kd[u] + Kui);
      } else {
        f_tmp = -INFINITY;     
      }
      pair = Pair{f_tmp, tid};
      res = BlockReduce(temp_storage.pair).Reduce(pair, cub::Max());
      int l = pair.key;
      math_t Kli = kernel[l * n_ws + tid];
      
      // check once more the final sign
       //update alpha
      if (threadIdx.x == u) // delta alpha_u
            tmp1 = y > 0 ? C - a : a;
      if (threadIdx.x == l) // delta alpha_l
            tmp2 = min(y > 0 ? a : C - a, (f_u - f) / (Kd[u] + Kd[l] - 2 * Kui));
      __syncthreads();
      math_t a_diff = min(tmp1, tmp2);
      
      if (threadIdx.x == u) a += a_diff * y;
      if (threadIdx.x == l) a -= a_diff * y;
      f += a_diff * (Kui - Kli);
    }
  }
}; // end namespace SVM
}; // end namespace ML
