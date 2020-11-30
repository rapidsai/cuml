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

#include <cuml/explainer/permutation_shap.hpp>

namespace ML {
namespace Explainer {

template <typename DataT, typename IdxT>
__global__ void _fused_tile_scatter_pe(DataT* vec, const DataT* bg,
                                       IdxT nrows_bg, IdxT ncols,
                                       const DataT* obs, IdxT* idx, IdxT len_bg,
                                       IdxT sc_size, bool row_major) {
  // kernel that actually does the scattering as described in the
  // descriptions of `permutation_dataset` and `shap_main_effect_dataset`
  IdxT tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < ncols * nrows_bg) {
    IdxT row, col, start, end;

    if (row_major) {
      row = tid / ncols;
      col = tid % ncols;
      start = (idx[col] + 1) * len_bg;
      end = start + sc_size * len_bg;

      if ((start <= row && row < end)) {
        vec[row * ncols + col] = obs[col];
      } else {
        vec[row * ncols + col] = bg[(row % len_bg) * ncols + col];
      }

    } else {
      col = tid / nrows_bg;
      row = tid % (len_bg);

      start = len_bg + idx[col] * len_bg;
      end = start + sc_size * len_bg;

      if ((start <= (row) && (row) < end)) {
        vec[tid] = obs[col];
      } else {
        vec[tid] = bg[row + len_bg * col];
      }
    }
  }
}

template <typename DataT, typename IdxT>
void permutation_shap_dataset_impl(const raft::handle_t& handle, DataT* out,
                                   const DataT* background, IdxT nrows_bg,
                                   IdxT ncols, const DataT* row, IdxT* idx,
                                   bool row_major) {
  const auto& handle_impl = handle;
  cudaStream_t stream = handle_impl.get_stream();

  IdxT total_num_elements = (2 * ncols * nrows_bg + nrows_bg) * ncols;

  constexpr IdxT Nthreads = 512;

  IdxT nblks = (total_num_elements + Nthreads - 1) / Nthreads;

  _fused_tile_scatter_pe<<<nblks, Nthreads, 0, stream>>>(
    out, background, total_num_elements / ncols, ncols, row, idx, nrows_bg,
    ncols, row_major);

  CUDA_CHECK(cudaPeekAtLastError());
}

void permutation_shap_dataset(const raft::handle_t& handle, float* out,
                              const float* background, int nrows_bg, int ncols,
                              const float* row, int* idx, bool row_major) {
  permutation_shap_dataset_impl(handle, out, background, nrows_bg, ncols, row,
                                idx, row_major);
}

void permutation_shap_dataset(const raft::handle_t& handle, double* out,
                              const double* background, int nrows_bg, int ncols,
                              const double* row, int* idx, bool row_major) {
  permutation_shap_dataset_impl(handle, out, background, nrows_bg, ncols, row,
                                idx, row_major);
}

template <typename DataT, typename IdxT>
void shap_main_effect_dataset_impl(const raft::handle_t& handle, DataT* out,
                                   const DataT* background, IdxT nrows_bg,
                                   IdxT ncols, const DataT* row, IdxT* idx,
                                   bool row_major) {
  const auto& handle_impl = handle;
  cudaStream_t stream = handle_impl.get_stream();

  IdxT total_num_elements = (nrows_bg * ncols + nrows_bg) * ncols;

  constexpr IdxT nthreads = 512;

  IdxT nblks = (total_num_elements + nthreads - 1) / nthreads;

  _fused_tile_scatter_pe<<<nblks, nthreads, 0, stream>>>(
    out, background, total_num_elements / ncols, ncols, row, idx, nrows_bg, 1,
    row_major);

  CUDA_CHECK(cudaPeekAtLastError());
}

void shap_main_effect_dataset(const raft::handle_t& handle, float* out,
                              const float* background, int nrows_bg, int ncols,
                              const float* row, int* idx, bool row_major) {
  shap_main_effect_dataset_impl(handle, out, background, nrows_bg, ncols, row,
                                idx, row_major);
}

void shap_main_effect_dataset(const raft::handle_t& handle, double* out,
                              const double* background, int nrows_bg, int ncols,
                              const double* row, int* idx, bool row_major) {
  shap_main_effect_dataset_impl(handle, out, background, nrows_bg, ncols, row,
                                idx, row_major);
}

template <typename DataT, typename IdxT>
__global__ void update_perm_shap_values_kernel(DataT* output,
                                               const DataT* input,
                                               const IdxT ncols,
                                               const IdxT* idx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < ncols) {
    DataT result = output[idx[tid]];
    result += input[tid + 1] - input[tid];
    result += input[tid + ncols] - input[tid + ncols + 1];
    output[idx[tid]] = result;
  }
}

template <typename DataT, typename IdxT>
void update_perm_shap_values_impl(const raft::handle_t& handle,
                                  DataT* shap_values, const DataT* y_hat,
                                  const IdxT ncols, const IdxT* idx) {
  const auto& handle_impl = handle;
  cudaStream_t stream = handle_impl.get_stream();

  constexpr IdxT nthreads = 512;

  IdxT nblks = ncols / nthreads + 1;

  update_perm_shap_values_kernel<<<nblks, nthreads, 0, 0>>>(shap_values, y_hat,
                                                            ncols, idx);

  CUDA_CHECK(cudaPeekAtLastError());
}

void update_perm_shap_values(const raft::handle_t& handle, float* shap_values,
                             const float* y_hat, const int ncols,
                             const int* idx) {
  update_perm_shap_values_impl(handle, shap_values, y_hat, ncols, idx);
}

void update_perm_shap_values(const raft::handle_t& handle, double* shap_values,
                             const double* y_hat, const int ncols,
                             const int* idx) {
  update_perm_shap_values_impl(handle, shap_values, y_hat, ncols, idx);
}

}  // namespace Explainer
}  // namespace ML
