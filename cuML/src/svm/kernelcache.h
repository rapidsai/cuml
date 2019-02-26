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
#include <iostream>
#include <stdlib.h>

namespace ML {
namespace SVM {

using namespace MLCommon;
/*
template<typname math_t>
__global__ void fill_x_ws(math_t* x, int n_rows, int n_cols, math_t* x_ws, int n_ws,
                          int *ws_idx, int *ws_idx_prev, int n_ws_prev) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  int i = tid % n_rows; // row idx
  int k = tid / n_rows; // col idx
  
  if i in ws_idx_prev:
    i_target = 
    //x_ws[k*n_rows + i_target] = x[x*n_rows + i] we do not need to copy the x value
    // only the kernel value
    tile[i_target,?] = tile_old[i,k]
  else:
    just
}

/// idx_set_prev and idx_set can have an overlap. 
/// Calculate how to remap the indices of idx_set_prev, in order to get 
/// all the existing indices to the beginning of idx_set
__global__ void calc_ws_remap(int *idx_set, int n_ws, int *idx_set_prev, 
    int * indices_kept, int *idx_remap) {
  typedef cub::BlockScan<int, 128> BlockScan;
  __shared__ int idx[1024];
  __shared  typename BlockScan::TempStorage temp_storage;
  __shared__ n_idx_kept;
  int tid = threadIdx.x;
  idx[tid] = idx_set_prev[tid];
  int idx_new = idx_set[tid];
  int remap_idx;
  
  int found = 0;
  for (i=0; i<1024; i++) {
    if (idx_new == idx[tid]) {
      found = 1;
      remap_idx = i;
    }
  }
  int sum_prev;
  // Collectively compute the block-wide exclusive prefix sum
  BlockScan(temp_storage).ExclusiveSum(found, sum_prev);
  if (tid == blockDim.x-1) {
    n_idx_kept = found + sum_prev;
    *indices_kept = n_idx_kept;
  }
  __synchthreads();
  
  remap[tid] = found ? sum_prev : -1;
  if (found) idx_set[sum_prev] = idx_new;
  
  int not_found = 1 - found;
  BlockScan(temp_storage).ExclusiveSum(not_found, sum_prev);
  if (not_found) idx_set[n_idx_kept + sum_prev] = idx_new;
}
*/

template <typename math_t>
__global__ void collect_rows(math_t *x, int n_rows, int n_cols, 
                              math_t *x_ws, int n_ws, int *ws_idx)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ws_row = tid % n_ws; // row idx
  if (ws_row < n_ws && tid < n_ws * n_cols) {
    int x_row = ws_idx[ws_row]; // row in the original matrix
    int col = tid / n_ws; // col idx
    //if (x_row + col * n_rows < n_rows*n_cols) {
      x_ws[tid] = x[x_row + col * n_rows];  
    //} else {
    //  printf("wrong memory access %d %d %d %d", tid, ws_row, x_row, col);
    //}
  }
}

/**
* Buffer to store a kernel tile.
*/ 
template<typename math_t>
class KernelCache {
  math_t *x, *host_x; 
  math_t *x_ws; // feature vectors in the current working set
  int *ws_idx_prev;
  int n_ws_prev = 0;
  int n_rows;
  int n_cols;
  int n_ws;
  
  math_t *tile = nullptr;
  cublasHandle_t cublas_handle;
  
public:
  KernelCache(math_t *x, int n_rows, int n_cols, int n_ws, cublasHandle_t cublas_handle) 
    : x(x), n_rows(n_rows), n_cols(n_cols), n_ws(n_ws), cublas_handle(cublas_handle)
  {
    allocate(x_ws, n_ws*n_cols);
    allocate(host_x, n_rows*n_cols);
    allocate(tile, n_rows*n_ws);
    allocate(ws_idx_prev, n_ws);
  };
  
  ~KernelCache() {
    CUDA_CHECK(cudaFree(tile));
    CUDA_CHECK(cudaFree(x_ws));
    CUDA_CHECK(cudaFree(host_x));
    CUDA_CHECK(cudaFree(ws_idx_prev));
  };
  
  math_t* GetTile(int *ws_idx) {
    // implementing only linear kernel so far
    // we need to gather ws_idx rows

    //x_ws = x; 
    
    const int TPB=256;
    collect_rows<<<ceildiv(n_ws*n_cols,TPB), TPB>>>(x, n_rows, n_cols, x_ws, n_ws, ws_idx);
    CUDA_CHECK(cudaPeekAtLastError());
    std::cout<<"input:\n";
//    updateHost(host_x, x_ws, n_ws*n_cols);
    for (int i=0; i<n_ws*n_cols; i++) {
//      std::cout<<host_x[i]<<" ";
    }
    std::cout<<"\n";
     
    //calc_ws_remap<<<n_ws,n_ws>>>(w_idx_set, n_ws, ws_idx_set_prev, idx_remap);
    //fill_x_ws<<<ceildiv(n_rows*n_cols,TPB), TPB>>>(x, n_rows, n_cols, x_ws, n_ws, ws_idx, ws_idx_prev, n_ws_prev);
    //calculate kernel function values for indices in ws_idx
    LinAlg::gemm(x_ws, n_ws, n_cols, x, tile, n_ws, n_rows, CUBLAS_OP_N,
          CUBLAS_OP_T, math_t(1.0), math_t(0.0), cublas_handle) ;
    
    n_ws_prev = n_ws;
    copy(ws_idx_prev, ws_idx, n_ws);
    return tile;
  }
};

}; // end namespace SVM 
}; // end namespace ML
