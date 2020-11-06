#pragma once
// data [T]ype, reduction [R]adix
template <int R, typename T>
// per_thread holds one value per thread in shared memory
// set_size is the number of indendent reductions
// per_thread values are ordered such that the stride is 1 for values belonging to the same set
// and set_size for values that are to be added together
// n_sets is the size of each individual reduction, or the number of sets to be reduced to one set
__device__ T multi_reduction(T* per_thread, const int set_size, int n_sets) {
  // reduce per-thread summand into a single per-block value, valid only for
  // threads 0..set_size-1
  T acc;  // accumulator
  if (threadIdx.x < set_size * n_sets) acc = per_thread[threadIdx.x];
  while (n_sets >= R) {
    if (threadIdx.x < n_sets / R * set_size) {
      // the following gets forwarded to the next iteration:
      // skip the (this time) unreduceable modulus [n_sets % R * set_size]
      // skip the [n_sets / R * set_size] sum destination values - set_id
      // starts with 1 to include this
#pragma unroll(R)
      for (int set_grp = 1; set_grp < R; ++set_grp)
        acc += per_thread[(n_sets % R + n_sets / R * set_grp) * set_size +
                          threadIdx.x];
      per_thread[threadIdx.x] = acc;
    }
    __syncthreads();
    n_sets = n_sets / R + n_sets % R;
  }  // now n_sets is [1..R-1]
  if (threadIdx.x < set_size) {
// the corrent number is R - 2, but the compiler does not detect that the loop is moot for R == 2
// leading to a warning
#pragma unroll(R - 1)
    for (int set_id = n_sets - 1; set_id; --set_id)
      acc += per_thread[threadIdx.x + set_id * set_size];
  }
  if (n_sets > 1) __syncthreads();  // free up per_thread[]
  return acc;
}
