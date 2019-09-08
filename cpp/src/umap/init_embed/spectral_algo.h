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

#include "common/device_buffer.hpp"
#include "umap/umapparams.h"

#include "sparse/coo.h"

#include "linalg/transpose.h"

#include <iostream>
#include "spectral/spectral.hpp"

namespace UMAPAlgo {

namespace InitEmbed {

namespace SpectralInit {

using namespace ML;

/**
   * Performs a spectral layout initialization
   */
template <typename T>
void launcher(const cumlHandle &handle, const T *X, int n, int d,
              const long *knn_indices, const T *knn_dists,
              MLCommon::Sparse::COO<float> *coo, UMAPParams *params,
              T *embedding) {

  ASSERT(n > params->n_components, "Spectral layout requires n_samples > n_components");

  MLCommon::device_buffer<T> tmp_storage(
    handle.getDeviceAllocator(), handle.getStream(), n * params->n_components);

  Spectral::fit_embedding(handle, coo->rows, coo->cols, coo->vals, coo->nnz, n,
                          params->n_components, tmp_storage.data());

  MLCommon::LinAlg::transpose(
    embedding, tmp_storage.data(), params->n_components, n,
    handle.getImpl().getCublasHandle(), handle.getStream());

  CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace SpectralInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
