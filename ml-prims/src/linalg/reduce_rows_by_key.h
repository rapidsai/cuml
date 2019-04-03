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

#define MAX_BLOCKS 65535u

#include "cuda_utils.h"
#include <cub/cub.cuh>
#include <limits>
#include <stdlib.h>


namespace MLCommon {
namespace LinAlg {

//
// Small helper function to convert from int->char and char->int 
// Transform ncols*nrows read of int in 2*nrows reads of int + ncols*rows reads of chars
//

template<typename IteratorT1,
	 typename IteratorT2>
void __global__ convert_array_kernel(IteratorT1 dst, IteratorT2 src, int n) {
  for(int idx = blockDim.x * blockIdx.x + threadIdx.x;
      idx < n;
      idx += gridDim.x * blockDim.x) {
    dst[idx] = src[idx];
  }
}


template<typename IteratorT1,
	 typename IteratorT2>
void convert_array(IteratorT1 dst, IteratorT2 src, int n, cudaStream_t st) {
  dim3 grid, block;
  block.x = 256;

  grid.x  = (n + block.x - 1)/block.x;
  grid.x = std::min(grid.x, MAX_BLOCKS);
    
  convert_array_kernel<<<grid,block,0,st>>>(dst, src, n);
}


template <typename T>
struct quad
{
  T x,y,z,w;
};
//
// Functor for reduce by key, small k
//
template <typename T>
struct quadSum
{
  __host__ __device__ __forceinline__ quad<T> operator()(const quad<T> &a, const quad<T> &b) const
  {
    // wasting a double4..
    quad<T> c;
    c.x = a.x + b.x; 
    c.y = a.y + b.y; 
    c.z = a.z + b.z; 
    c.w = a.w + b.w; 

    return c;
  }
};

//
// Reduce by keys
// We need to sum each dimension by labels
// The labels are not adjacent
//

//
// Reduce by keys - for keys <= 4
//

#define SUM_ROWS_SMALL_K_DIMX 256
#define SUM_ROWS_BY_KEY_SMALL_K_MAX_K 4 
template <typename DataIteratorT>
__launch_bounds__(SUM_ROWS_SMALL_K_DIMX, 4)
  __global__ void sum_rows_by_key_small_nkeys_kernel(const DataIteratorT d_A,
						     int lda,
						     char *d_keys,
						     int nrows,
						     int ncols,
						     int nkeys,
						     DataIteratorT d_sums) {
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
  typedef cub::BlockReduce<quad<DataType>, SUM_ROWS_SMALL_K_DIMX> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for(int idim = blockIdx.y;
      idim < ncols;
      idim += gridDim.y) {

    if(idim != blockIdx.y)
      __syncthreads(); // we're reusing temp_storage

    // threadIdx.x stores partial sum for current dim and key=threadIdx.x in this reg
    quad<DataType> thread_sums;
    thread_sums.x = 0.0;
    thread_sums.y = 0.0;
    thread_sums.z = 0.0;
    thread_sums.w = 0.0;

    // May use vectorized load - not necessary for doubles
    for(int block_offset_irow = blockIdx.x * blockDim.x;
	block_offset_irow < nrows; // we will syncthreads() inside the loop, no CTA divergence
	block_offset_irow += blockDim.x * gridDim.x) {

      int irow = block_offset_irow + threadIdx.x;
      DataType val = (irow < nrows) ? d_A[irow * lda + idim] : 0.0;
      // we are not reusing the keys - after profiling 
      // d_keys is mainly loaded from L2, and this kernel is DRAM BW bounded
      // (experimentation gave a 10% speed up - not worth the many code lines added)
      int row_key = (irow < nrows) ? d_keys[irow] : -1;

      thread_sums.x += (row_key == 0) ? val : 0.0;
      thread_sums.y += (row_key == 1) ? val : 0.0;
      thread_sums.z += (row_key == 2) ? val : 0.0;
      thread_sums.w += (row_key == 3) ? val : 0.0;
    }

    // End of column
    // Saving local sums back to global mem

    // Strided access


    // Reducing by key
    thread_sums = BlockReduce(temp_storage).Reduce(thread_sums, quadSum<DataType>());

    if(threadIdx.x < 32) {
      // We only need 4
      thread_sums = cub::ShuffleIndex<32>(thread_sums, 0, 0xffffffff);
      if (threadIdx.x < nkeys) {
	if(threadIdx.x == 0)    myAtomicAdd(&d_sums[threadIdx.x*ncols + idim], thread_sums.x);
	if(threadIdx.x == 1)    myAtomicAdd(&d_sums[threadIdx.x*ncols + idim], thread_sums.y);
	if(threadIdx.x == 2)    myAtomicAdd(&d_sums[threadIdx.x*ncols + idim], thread_sums.z);
	if(threadIdx.x == 3)    myAtomicAdd(&d_sums[threadIdx.x*ncols + idim], thread_sums.w);
      }
    }
  }

}


template <typename DataIteratorT>
void sum_rows_by_key_small_nkeys(const DataIteratorT d_A,
				 int lda,
				 char *d_keys,
				 int nrows,
				 int ncols,
				 int nkeys,
				 DataIteratorT d_sums,
				 cudaStream_t st) {
  dim3 grid,block;
  block.x = SUM_ROWS_SMALL_K_DIMX;
  block.y = 1; // Necessary

  grid.x = (nrows + block.x -1) / grid.x;
  grid.x = std::min(grid.x, 32u); 
  grid.y = ncols; 
  grid.y = std::min(grid.y, MAX_BLOCKS);
  sum_rows_by_key_small_nkeys_kernel<<<grid,block,0,st>>>(d_A, lda, d_keys, nrows, ncols, nkeys, d_sums);   
}


//
// Reduce by keys - large number of keys
// Computing a "weigthed histogram" with local histograms in smem
// Keeping it simple - not optimized
//

#define SUM_ROWS_BY_KEY_LARGE_K_MAX_K 1024 
template <typename DataIteratorT,
	  typename KeysIteratorT>
__global__ void sum_rows_by_key_large_nkeys_kernel_colmajor(const DataIteratorT d_A,
							    int lda,
							    KeysIteratorT d_keys,
							    int nrows,
							    int ncols,
							    int key_offset,
							    int nkeys,
							    DataIteratorT d_sums) {
  typedef typename std::iterator_traits<KeysIteratorT>::value_type KeyType;
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
  __shared__ DataType local_sums[SUM_ROWS_BY_KEY_LARGE_K_MAX_K];

  for(int local_key=threadIdx.x; local_key<nkeys; local_key+=blockDim.x)
    local_sums[local_key] = 0.0;

  for(int idim = blockIdx.y;
      idim < ncols;
      idim += gridDim.y) {
            
    __syncthreads(); // local_sums
               
    // At this point local_sums if full of zeros 

    for(int irow = blockIdx.x * blockDim.x + threadIdx.x;
	irow < nrows;
	irow += blockDim.x * gridDim.x) {
      // Branch div in this loop - not an issue with current code 
      DataType val  = d_A[idim*lda + irow];
      int local_key = d_keys[irow] - key_offset;

      // We could load next val here
      myAtomicAdd(&local_sums[local_key], val);
    }
                
    __syncthreads(); // local_sums
                
    for(int local_key=threadIdx.x; local_key<nkeys; local_key+=blockDim.x) {
      DataType local_sum = local_sums[local_key];

      if(local_sum != 0.0) {
	KeyType global_key = key_offset + local_key;
	myAtomicAdd(&d_sums[global_key*ncols + idim], local_sum);
	local_sums[local_key] = 0.0;
      }
    }
  }
        
}

template <typename DataIteratorT,
	  typename KeysIteratorT>
void sum_rows_by_key_large_nkeys_colmajor(const DataIteratorT d_A,
					  int lda, 
					  KeysIteratorT d_keys,
					  int nrows,
					  int ncols,
					  int key_offset, 
					  int nkeys,
					  DataIteratorT d_sums,
					  cudaStream_t st) {
  dim3 grid,block;
  block.x = SUM_ROWS_SMALL_K_DIMX;
  block.y = 1; // Necessary

  grid.x = (nrows + block.x -1) / grid.x;
  grid.x = std::min(grid.x, 32u);
  grid.y = ncols; 
  grid.y = std::min(grid.y, MAX_BLOCKS);
  sum_rows_by_key_large_nkeys_kernel_colmajor<<<grid,block,0,st>>>(d_A, lda, d_keys, nrows, ncols, key_offset, nkeys, d_sums);   
}

#define RRBK_SHMEM_SZ 32
//#define RRBK_SHMEM
template <typename DataIteratorT,
	  typename KeysIteratorT>
__global__ void sum_rows_by_key_large_nkeys_kernel_rowmajor(const DataIteratorT d_A,
							    int lda,
							    KeysIteratorT d_keys,
							    int nrows,
							    int ncols,
							    int key_offset,
							    int nkeys,
							    DataIteratorT d_sums) {
  typedef typename std::iterator_traits<KeysIteratorT>::value_type KeyType;
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;

#ifdef RRBK_SHMEM
  __shared__ KeyType sh_keys[RRBK_SHMEM_SZ];
#endif
  int rows_per_partition = nrows/gridDim.z+1;
  int start_row = blockIdx.z*rows_per_partition;
  int end_row = start_row + rows_per_partition;
  end_row = end_row > nrows ? nrows : end_row;

  KeyType this_key = blockIdx.y;
  if (this_key >= nkeys) return;
  int this_col = threadIdx.x + blockIdx.x*blockDim.x;
  if (this_col >= ncols) return;
        
  DataType sum = 0.0;

#ifdef RRBK_SHMEM
  int sh_key_inx = 0;
#endif
  for (int r = start_row; r < end_row; r++) {
#ifdef RRBK_SHMEM
    if (0 == sh_key_inx%RRBK_SHMEM_SZ) {
      for (int x=threadIdx.x;x<RRBK_SHMEM_SZ;x+=blockDim.x)
	sh_keys[x] = d_keys[r+x];
      __syncthreads();
    }
    if (sh_keys[sh_key_inx] != this_key) continue; //No divergence since this_key is the
    // same for the whole block
    sh_key_inx++;
#else
    if (d_keys[r] != this_key) continue; //No divergence since this_key is the
    // same for the whole block
#endif
    //if ((end_row-start_row) / (r-start_row) != this_key) continue;
    sum += __ldcg(&d_A[r*lda+this_col]);
  }
        
  if (sum != 0.0) myAtomicAdd(&d_sums[this_key*ncols+this_col], sum);
}

template <typename DataIteratorT,
	  typename KeysIteratorT>
void sum_rows_by_key_large_nkeys_rowmajor( const DataIteratorT d_A,
					   int lda, 
					   KeysIteratorT d_keys,
					   int nrows,
					   int ncols,
					   int key_offset, 
					   int nkeys,
					   DataIteratorT d_sums,
					   cudaStream_t st)  {
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;

  // x-dim refers to the column in the input data
  // y-dim refers to the key
  // z-dim refers to a partitioning of the rows among the threadblocks
  dim3 grid, block;
  block.x = 256; //Adjust me!
  block.y = 1; //Don't adjust me!
  grid.x = (ncols+block.x-1)/block.x;
  grid.y = nkeys;
  grid.z = std::max(40960000/nkeys/ncols, (int)1); //Adjust me!
  grid.z = std::min(grid.z, (unsigned int)nrows);
  grid.z = std::min(grid.z, MAX_BLOCKS);
  //std::cout << "block = " << block.x << ", " << block.y << std::endl;
  //std::cout << "grid = " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
  cudaMemset(d_sums, 0, sizeof(DataType)*nkeys*ncols);
  sum_rows_by_key_large_nkeys_kernel_rowmajor<<<grid,block,0,st>>>(d_A, lda, d_keys, nrows, ncols, key_offset, nkeys, d_sums);   
}

 
    
/**
 * @brief Computes the reduction of matrix rows for each given key 
 * @tparam DataIteratorT Random-access iterator type, for reading input matrix (may be a simple pointer type)
 * @tparam KeysIteratorT Random-access iterator type, for reading input keys (may be a simple pointer type)
 * @param[in]  d_A         Input data array (lda x nrows)
 * @param[in]  lda         Real row size for input data, d_A
 * @param[in]  d_keys      Keys for each row (1 x nrows)
 * @param      d_keys_char Scratch memory for conversion of keys to char
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys 
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
 */
template <typename DataIteratorT,
	  typename KeysIteratorT>
void reduce_rows_by_key(const DataIteratorT d_A,
			int lda,
			const KeysIteratorT d_keys, 
			char *d_keys_char,
			int nrows,
			int ncols,
			int nkeys, 
			DataIteratorT d_sums,
			cudaStream_t stream) {
  typedef typename std::iterator_traits<KeysIteratorT>::value_type KeyType;
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
      
  // Following kernel needs memset
  cudaMemsetAsync(d_sums, 0, ncols * nkeys * sizeof(DataType), stream);
   

  if(nkeys <= SUM_ROWS_BY_KEY_SMALL_K_MAX_K) {
    // sum_rows_by_key_small_k is BW bounded. d_keys is loaded ncols time - avoiding wasting BW
    // with doubles we have ~20% speed up - with floats we can hope something around 2x
    // Converting d_keys to char
    convert_array(d_keys_char, d_keys, nrows, stream);
    sum_rows_by_key_small_nkeys(d_A, lda, d_keys_char, nrows, ncols, nkeys, d_sums, stream);
  } else {
    for(KeyType key_offset = 0;
	key_offset < nkeys;
	key_offset += SUM_ROWS_BY_KEY_LARGE_K_MAX_K) {
      KeyType this_call_nkeys = std::min(SUM_ROWS_BY_KEY_LARGE_K_MAX_K, nkeys);
      sum_rows_by_key_large_nkeys_rowmajor(d_A, lda, d_keys, nrows, ncols, key_offset, this_call_nkeys, d_sums, stream);
    }
  }
    
}

}; // end namespace LinAlg
}; // end namespace MLCommon
