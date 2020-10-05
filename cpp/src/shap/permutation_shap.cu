/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuml/datasets/make_permutation.hpp>

namespace ML {
namespace Datasets {

template <typename DataT, typename IdxT>
__global__ void _fused_tile_scatter_pr(DataT *vec, int n, int m,
                                       DataT *obs,
                                       int *idx,
                                       int len_bg,
                                       bool rowMajor){
    // kernel does the scattering of the obs column with one thread
    // per entry of the obs value. It is not currently used but here
    // for debugging purposes and in case the one thread per vec entry kernel
    // below would have problems with very large matrix sizes
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < m*n){
        int row, col, start, end;


        if(rowMajor){
            row = tid / m, col = tid % m;
            for(int curval=0; curval < m; curval++){
                start = (curval+1)*len_bg;
                end = start + m*len_bg;

                if((start <= row && row < end && col == idx[curval])){
                    vec[row * m + col] = obs[idx[curval]];
                }
            }
        }else{
            for(int curval=0; curval < m ; curval++){
                start = (curval + 1) * len_bg + idx[curval] * n;
                end = start + m*len_bg;

                if((start <= tid && tid < end)){
                    vec[tid] = obs[idx[curval]];
                }
            }

        }
    }
}

template <typename DataT, typename IdxT>
__global__ void _fused_tile_scatter_pe(DataT *vec,
                                       DataT *bg,
                                       IdxT n,
                                       IdxT m,
                                       DataT *obs,
                                       IdxT *idx,
                                       IdxT len_bg,
                                       IdxT sc_size,
                                       bool rowMajor){
    // kernel that actually does the scattering as described in the
    // descriptions of `permutation_dataset` and `main_effect_dataset`
    IdxT tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < m*n){
        // printf("tid, m, n: %d %d %d\n", tid, m, n);
        IdxT row, col, start, end;

        if(rowMajor){
            row = tid / m;
            col = tid % m;
            start = (idx[col] + 1) * len_bg;
            end = start + sc_size * len_bg;

            if((start <= row && row < end)){
                vec[row * m + col] = obs[col];
            } else {
                vec[row * m + col] = bg[(row % len_bg) * m + col];
            }

        } else {
            col = tid / n;

            start = len_bg + idx[col] * len_bg;
            end = start + sc_size * len_bg;

            if((start <= (tid % n) && (tid % n) < end)){
                vec[tid] = obs[col];
            } else {
                vec[tid] = bg[tid % (len_bg) + len_bg * col];
            }

        }

    }
}


template <typename DataT, typename IdxT>
void permutation_dataset_impl(const raft::handle_t& handle, DataT* out,
                           DataT* background, IdxT n_rows, IdxT n_cols,
                           DataT* row, IdxT* idx, bool rowMajor) {
    const auto& handle_impl = handle;
    cudaStream_t stream = handle_impl.get_stream();

    IdxT total_num_elements = (2 * n_cols * n_rows + n_rows) * n_cols;

    constexpr IdxT Nthreads = 256;

    IdxT nblks = (total_num_elements + Nthreads -1)/Nthreads;

    _fused_tile_scatter_pe<<<nblks, Nthreads, 0, stream>>>(
        out, background, total_num_elements/n_cols, n_cols, row, idx,
        n_rows, n_cols, rowMajor);

    CUDA_CHECK(cudaPeekAtLastError());

}


void permutation_dataset(const raft::handle_t& handle, float* out,
                      float* background, int n_rows, int n_cols,
                      float* row, int* idx, bool rowMajor){
    permutation_dataset_impl(handle, out, background, n_rows, n_cols,
                          row, idx, rowMajor);
}


void permutation_dataset(const raft::handle_t& handle, double* out,
                      double* background, int n_rows, int n_cols,
                      double* row, int* idx, bool rowMajor){
    make_permutation_impl(handle, out, background, n_rows, n_cols,
                          row, idx, rowMajor);
}



template <typename DataT, typename IdxT>
void main_effect_dataset_impl(const raft::handle_t& handle, DataT* out,
                               DataT* background, IdxT n_rows, IdxT n_cols,
                               DataT* row, IdxT* idx, bool rowMajor) {
    const auto& handle_impl = handle;
    cudaStream_t stream = handle_impl.get_stream();

    IdxT total_num_elements = (n_rows * n_cols + n_rows) * n_cols;

    constexpr IdxT Nthreads = 256;

    IdxT nblks = (total_num_elements + Nthreads -1)/Nthreads;

    _fused_tile_scatter_pe<<<nblks, Nthreads, 0, stream>>>(
        out, background, total_num_elements/n_cols, n_cols, row, idx,
        n_rows, 1, rowMajor);

    CUDA_CHECK(cudaPeekAtLastError());

}



void main_effect_dataset(const raft::handle_t& handle, float* out,
                          float* background, int n_rows, int n_cols,
                          float* row, int* idx, bool rowMajor){
    main_effect_dataset_impl(handle, out, background, n_rows, n_cols,
                              row, idx, rowMajor);
}

void main_effect_dataset(const raft::handle_t& handle, double* out,
                          double* background, int n_rows, int n_cols,
                          double* row, int* idx, bool rowMajor){
    main_effect_dataset_impl(handle, out, background, n_rows, n_cols,
                              row, idx, rowMajor);
}

}  // namespace Datasets
}  // namespace ML
