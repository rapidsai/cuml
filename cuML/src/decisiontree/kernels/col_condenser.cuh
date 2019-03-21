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
#include <thrust/sort.h>

template <class type>
__global__ void get_sampled_column_kernel(const type *column,type *outcolumn,unsigned int* rowids,const int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N)
		{
			int index = rowids[tid];
			outcolumn[tid] = column[index];
		}
	return;
}

void get_sampled_column(const float *column,float *outcolumn,unsigned int* rowids,const int n_sampled_rows,const cudaStream_t stream = 0)
{
	get_sampled_column_kernel<float><<<(int)(n_sampled_rows / 128) + 1,128,0,stream>>>(column,outcolumn,rowids,n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}


void get_sampled_labels(const int *labels,int *outlabels,unsigned int* rowids,const int n_sampled_rows,const cudaStream_t stream = 0)
{
	get_sampled_column_kernel<int><<<(int)(n_sampled_rows / 128) + 1,128,0,stream>>>(labels,outlabels,rowids,n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return;
}


__device__ __forceinline__ float atomic_max_fp(float* address, float val) {

    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ __forceinline__ float atomic_min_fp(float* address, float val) {

    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

/* Merged kernel: gets sampled column and also produces min and max values. */
__global__ void get_sampled_column_minmax_kernel(const float *column, float *outcolumn, const unsigned int* rowids, float * col_min_max, const int N) {

	__shared__ float shmem_min_max[2];
	float column_val;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {
		int index = rowids[tid];
		column_val = column[index];
		outcolumn[tid] = column_val;

		//  Initialize min max values in shared memory
		if (threadIdx.x == 0) { 
			shmem_min_max[0] = column_val;
			shmem_min_max[1] = column_val;
		}

		// Initialize min max values in global memory. 
		if (tid == 0) {
			col_min_max[0] = column_val;
			col_min_max[1] = column_val;
		}
	}

	__syncthreads();

	// Min - max reduction within each block.
	if (tid < N) {
		atomic_min_fp(&shmem_min_max[0], column_val);
		atomic_max_fp(&shmem_min_max[1], column_val);
	}

	__syncthreads();

	// Min - max reduction across blocks.
	if (threadIdx.x == 0) {
		atomic_min_fp(&col_min_max[0], shmem_min_max[0]);
		atomic_max_fp(&col_min_max[1], shmem_min_max[1]);
	}

}


void get_sampled_column_minmax(const float *column, float *outcolumn, unsigned int* rowids, const int n_sampled_rows,  TemporaryMemory* tempmem) {

	ASSERT(n_sampled_rows != 0, "Column sampling for empty column\n");
	get_sampled_column_minmax_kernel<<<(int)(n_sampled_rows / 128) + 1, 128, 0, tempmem->stream>>>(column, outcolumn, rowids, tempmem->d_min_max, n_sampled_rows);
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
}


