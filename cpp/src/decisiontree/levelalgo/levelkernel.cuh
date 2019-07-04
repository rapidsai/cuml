#pragma once

#define LEAF 0xFFFFFFFF
#define PUSHRIGHT 0x00000001
__device__ __forceinline__ bool check_condition(unsigned int local_flag,
                                                unsigned int nodectr,
                                                int batch_nodes) {
  if (local_flag == LEAF) return false;
  if (local_flag < nodectr * batch_nodes) return false;
  if (local_flag >= (nodectr + 1) * batch_nodes) return false;
  return true;
}
template <typename T>
__global__ void get_me_hist_kernel(
  const T* __restrict__ data, const int* __restrict__ labels,
  const unsigned int* __restrict__ flags, const int nrows, const int ncols,
  const int n_unique_labels, const int nbins, const int n_nodes,
  const T* __restrict__ quantile, int batch_nodes, unsigned int* histout) {
  extern __shared__ unsigned int shmemhist[];
  unsigned int local_flag;
  int local_label;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int max_node_itr = (int)(n_nodes / batch_nodes);
  if (n_nodes % batch_nodes != 0) max_node_itr++;

  if (tid < nrows) {
    local_flag = flags[tid];
    local_label = labels[tid];
  } else {
    local_flag = LEAF;
    local_label = -1;
  }

  for (unsigned int colid = 0; colid < ncols; colid++) {
    int nodecnt = batch_nodes;
    for (unsigned int nodectr = 0; nodectr < max_node_itr; nodectr++) {
      if (nodectr == (max_node_itr - 1) && ((n_nodes % batch_nodes) != 0))
        nodecnt = n_nodes % batch_nodes;

      for (unsigned int i = threadIdx.x; i < nbins * nodecnt * n_unique_labels;
           i += blockDim.x) {
        shmemhist[i] = 0;
      }
      __syncthreads();

      //Check if leaf
      if (check_condition(local_flag, nodectr, batch_nodes)) {
        T local_data = data[tid + colid * nrows];
        //Loop over nbins

#pragma unroll(8)
        for (unsigned int binid = 0; binid < nbins; binid++) {
          T quesval = quantile[colid * nbins + binid];
          if (local_data <= quesval) {
            unsigned int nodeoff =
              (local_flag % nodecnt) * nbins * n_unique_labels;
            atomicAdd(
              &shmemhist[nodeoff + binid * n_unique_labels + local_label], 1);
          }
        }
      }
      
      __syncthreads();
      for (unsigned int i = threadIdx.x; i < nbins * nodecnt * n_unique_labels;
           i += blockDim.x) {
        unsigned int offset = colid * nbins * n_nodes * n_unique_labels;
        offset += nodectr * nbins * batch_nodes * n_unique_labels;
        atomicAdd(&histout[offset + i], shmemhist[i]);
      }
      __syncthreads();
    }
  }
}

template <typename T>
__global__ void split_level_kernel(
  const T* __restrict__ data, const T* __restrict__ quantile,
  const unsigned int* __restrict__ split_col_index,
  const unsigned int* __restrict__ split_bin_index, const int nrows,
  const int ncols, const int nbins, const int n_nodes,
  const unsigned int* __restrict__ new_node_flags,
  unsigned int* __restrict__ flags) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int local_flag;

  if (tid < nrows) {
    local_flag = flags[tid];
  } else {
    local_flag = LEAF;
  }

  if (local_flag != LEAF) {
    unsigned int local_leaf_flag = new_node_flags[local_flag];
    if (local_leaf_flag != LEAF) {
      int colidx = split_col_index[local_flag];
      T quesval = quantile[colidx * nbins + split_bin_index[local_flag]];
      T local_data = data[colidx * nrows + tid];
      //The inverse comparision here to push right instead of left
      if (local_data <= quesval) {
        local_flag = local_leaf_flag << 1;
      } else {
        local_flag = (local_leaf_flag << 1) | PUSHRIGHT;
      }
    } else {
      local_flag = LEAF;
    }
    flags[tid] = local_flag;
  }
}
