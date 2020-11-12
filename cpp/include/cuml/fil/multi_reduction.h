#pragma once
#include <cuda_utils.cuh>
// data [T]ype, reduction [R]adix
template <int R, typename T>
/**
 per_thread holds one value per thread in shared memory
 n_groups is the number of indendent reductions
 per_thread values are ordered such that the stride is 1 for values belonging to the same group
 and n_groups for values that are to be added together
 n_values is the size of each individual reduction,
 that is the number of values to be reduced to a single value
*/
__device__ T multi_reduction(T* data, int n_groups, int n_values) {
  T acc = threadIdx.x < n_groups * n_values ? data[threadIdx.x] : T();
  while (n_values > 1) {
    // ceildiv(x, y) = (x + y - 1) / y, i.e. division with rounding upwards
    // n_targets is the number of values per group after the end of this iteration
    int n_targets = raft::ceildiv(n_values, R);
    if (threadIdx.x < n_targets * n_groups) {
#pragma unroll
      for (int i = 1; i < R; ++i) {
        int idx = threadIdx.x + i * n_targets * n_groups;
        if (idx < n_values * n_groups) acc += data[idx];
      }
      if (n_targets > 1) data[threadIdx.x] = acc;
    }
    n_values = n_targets;
    if (n_values > 1) __syncthreads();
  }
  return acc;
}
