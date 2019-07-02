#pragma once

#define LEAF 0xFFFFFFFF
#define PUSHRIGHT 0x00000001

template <typename T>
__global__ void get_me_hist_kernel(
  const T* __restrict__ data, const int* __restrict__ labels,
  const unsigned int* __restrict__ flags, const int nrows, const int ncols,
  const int n_unique_labels, const int nbins, const int n_nodes,
  const T* __restrict__ quantile, unsigned int* histout) {
  extern __shared__ char shmem[];
  unsigned int* shmemhist = (unsigned int*)(shmem);
  unsigned int local_flag;
  int local_label;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < nrows) {
    local_flag = flags[tid];
    local_label = labels[tid];
  } else {
    local_flag = LEAF;
    local_label = -1;
  }

  for (int colid = 0; colid < ncols; colid++) {
    for (int i = threadIdx.x; i < nbins * n_nodes * n_unique_labels;
         i += blockDim.x) {
      shmemhist[i] = 0;
    }
    __syncthreads();

    //Check if leaf
    if (local_flag != LEAF) {
      T local_data = data[tid + colid * nrows];
      //Loop over nbins

#pragma unroll(8)
      for (int binid = 0; binid < nbins; binid++) {
        T quesval = quantile[colid * nbins + binid];
        if (local_data <= quesval) {
          int nodeoff = local_flag * nbins * n_unique_labels;
          atomicAdd(&shmemhist[nodeoff + binid * n_unique_labels + local_label],
                    1);
        }
      }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < nbins * n_nodes * n_unique_labels;
         i += blockDim.x) {
      int offset = colid * nbins * n_nodes * n_unique_labels;
      atomicAdd(&histout[offset + i], shmemhist[i]);
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void split_level_kernel(const T* __restrict__ data,
                                   const T* __restrict__ quantile,
                                   const int* __restrict__ split_col_index,
                                   const int* __restrict__ split_bin_index,
                                   const int nrows, const int ncols,
                                   const int nbins, const int n_nodes,
                                   const bool* __restrict__ leaf_flag,
                                   unsigned int* __restrict__ flags) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int local_flag;

  if (tid < nrows) {
    local_flag = flags[tid];
  } else {
    local_flag = LEAF;
  }

  if (local_flag != LEAF) {
    bool local_leaf_flag = leaf_flag[local_flag];
    if(local_leaf_flag != true) {
      int colidx = split_col_index[local_flag];
      T quesval = quantile[colidx * nbins + split_bin_index[local_flag]];
      T local_data = data[colidx * nrows + tid];
      //The inverse comparision here to push right instead of left
      if (local_data <= quesval) {
	local_flag = local_flag << 1;
      } else {
	local_flag = (local_flag << 1) | PUSHRIGHT;
      }
    } else {
      local_flag = LEAF;
    }
    flags[tid] = local_flag;
  }
}
