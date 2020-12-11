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
__global__ void _fused_tile_scatter_pe(DataT* dataset, const DataT* background,
                                       IdxT nrows_dataset, IdxT ncols,
                                       const DataT* obs, IdxT* idx,
                                       IdxT nrows_background, IdxT sc_size,
                                       bool row_major) {
  // kernel that actually does the scattering as described in the
  // descriptions of `permutation_dataset` and `shap_main_effect_dataset`
  IdxT tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < ncols * nrows_dataset) {
    IdxT row, col, start, end;

    if (row_major) {
      row = tid / ncols;
      col = tid % ncols;
      start = (idx[col] + 1) * nrows_background;
      end = start + sc_size * nrows_background;

      if ((start <= row && row < end)) {
        dataset[row * ncols + col] = obs[col];
      } else {
        dataset[row * ncols + col] =
          background[(row % nrows_background) * ncols + col];
      }

    } else {
      col = tid / nrows_dataset;
      row = tid % nrows_dataset;

      start = nrows_background + idx[col] * nrows_background;
      end = start + sc_size * nrows_background;

      if ((start <= row && row < end)) {
        dataset[tid] = obs[col];
      } else {
        dataset[tid] = background[row + nrows_background * col];
      }
    }
  }
}

template <typename DataT, typename IdxT>
void permutation_shap_dataset_impl(const raft::handle_t& handle, DataT* dataset,
                                   const DataT* background,
                                   IdxT nrows_background, IdxT ncols,
                                   const DataT* row, IdxT* idx,
                                   bool row_major) {
  const auto& handle_impl = handle;
  cudaStream_t stream = handle_impl.get_stream();

  // we calculate the number of rows in the dataset and then multiply by 2 since
  // we are adding a forward and backward permutation (see docstring in header file)
  IdxT nrows_dataset = (2 * ncols * nrows_background + nrows_background);

  constexpr IdxT nthreads = 512;

  IdxT nblks = (nrows_dataset * ncols + nthreads - 1) / nthreads;

  _fused_tile_scatter_pe<<<nblks, nthreads, 0, stream>>>(
    dataset, background, nrows_dataset, ncols, row, idx, nrows_background,
    ncols, row_major);

  CUDA_CHECK(cudaPeekAtLastError());
}

void permutation_shap_dataset(const raft::handle_t& handle, float* dataset,
                              const float* background, int nrows_bg, int ncols,
                              const float* row, int* idx, bool row_major) {
  permutation_shap_dataset_impl(handle, dataset, background, nrows_bg, ncols,
                                row, idx, row_major);
}

void permutation_shap_dataset(const raft::handle_t& handle, double* dataset,
                              const double* background, int nrows_bg, int ncols,
                              const double* row, int* idx, bool row_major) {
  permutation_shap_dataset_impl(handle, dataset, background, nrows_bg, ncols,
                                row, idx, row_major);
}

template <typename DataT, typename IdxT>
void shap_main_effect_dataset_impl(const raft::handle_t& handle, DataT* dataset,
                                   const DataT* background, IdxT nrows_bg,
                                   IdxT ncols, const DataT* row, IdxT* idx,
                                   bool row_major) {
  const auto& handle_impl = handle;
  cudaStream_t stream = handle_impl.get_stream();

  // we calculate the number of rows in the dataset
  IdxT total_num_elements = (nrows_bg * ncols + nrows_bg) * ncols;

  constexpr IdxT nthreads = 512;

  IdxT nblks = (total_num_elements + nthreads - 1) / nthreads;

  _fused_tile_scatter_pe<<<nblks, nthreads, 0, stream>>>(
    dataset, background, total_num_elements / ncols, ncols, row, idx, nrows_bg,
    1, row_major);

  CUDA_CHECK(cudaPeekAtLastError());
}

void shap_main_effect_dataset(const raft::handle_t& handle, float* dataset,
                              const float* background, int nrows_bg, int ncols,
                              const float* row, int* idx, bool row_major) {
  shap_main_effect_dataset_impl(handle, dataset, background, nrows_bg, ncols,
                                row, idx, row_major);
}

void shap_main_effect_dataset(const raft::handle_t& handle, double* dataset,
                              const double* background, int nrows_bg, int ncols,
                              const double* row, int* idx, bool row_major) {
  shap_main_effect_dataset_impl(handle, dataset, background, nrows_bg, ncols,
                                row, idx, row_major);
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
