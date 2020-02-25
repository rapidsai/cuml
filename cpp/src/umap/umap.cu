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

#include <cuml/manifold/umapparams.h>
#include <cuml/manifold/umap.hpp>
#include "runner.h"

#include <iostream>

namespace ML {

static const int TPB_X = 256;

void transform(const cumlHandle &handle, float *X, int n, int d, float *orig_X,
               int orig_n, double *embedding, int embedding_n,
               UMAPParams *params, double *transformed) {
  UMAPAlgo::_transform<float, TPB_X>(handle, X, n, d, orig_X, orig_n, embedding,
                                     embedding_n, params, transformed);
}
void fit(const cumlHandle &handle,
         float *X,  // input matrix
         float *y,  // labels
         int n, int d, UMAPParams *params, double *embeddings) {
  UMAPAlgo::_fit<float, TPB_X>(handle, X, y, n, d, params, embeddings);
}

void fit(const cumlHandle &handle,
         float *X,  // input matrix
         int n,     // rows
         int d,     // cols
         UMAPParams *params, double *embeddings) {
  UMAPAlgo::_fit<float, TPB_X>(handle, X, n, d, params, embeddings);
}

void find_ab(const cumlHandle &handle, UMAPParams *params) {
  cudaStream_t stream = handle.getStream();
  auto d_alloc = handle.getDeviceAllocator();
  UMAPAlgo::find_ab(params, d_alloc, stream);
}
UMAP_API::UMAP_API(const cumlHandle &handle, UMAPParams *params)
  : params(params) {
  this->handle = const_cast<cumlHandle *>(&handle);
  orig_X = nullptr;
  orig_n = 0;
};

UMAP_API::~UMAP_API() {}

/**
 * Fits a UMAP model
 * @param X
 *        pointer to an array in row-major format (note: this will be col-major soon)
 * @param n
 *        n_samples in X
 * @param d
 *        d_features in X
 * @param embeddings
 *        an array to return the output embeddings of size (n_samples, n_components)
 */
void UMAP_API::fit(float *X, int n, int d, double *embeddings) {
  this->orig_X = X;
  this->orig_n = n;
  UMAPAlgo::_fit<float, TPB_X>(*this->handle, X, n, d, get_params(),
                               embeddings);
}

void UMAP_API::fit(float *X, float *y, int n, int d, double *embeddings) {
  this->orig_X = X;
  this->orig_n = n;

  UMAPAlgo::_fit<float, TPB_X>(*this->handle, X, y, n, d, get_params(),
                               embeddings);
}

/**
 * Project a set of X vectors into the embedding space.
 * @param X
 *        pointer to an array in row-major format (note: this will be col-major soon)
 * @param n
 *        n_samples in X
 * @param d
 *        d_features in X
 * @param embedding
 *        pointer to embedding array of size (embedding_n, n_components) that has been created with fit()
 * @param embedding_n
 *        n_samples in embedding array
 * @param out
 *        pointer to array for storing output embeddings (n, n_components)
 */
void UMAP_API::transform(float *X, int n, int d, double *embedding,
                         int embedding_n, double *out) {
  UMAPAlgo::_transform<float, TPB_X>(*this->handle, X, n, d, this->orig_X,
                                     this->orig_n, embedding, embedding_n,
                                     get_params(), out);
}

/**
 * Get the UMAPParams instance
 */
UMAPParams *UMAP_API::get_params() { return this->params; }
}  // namespace ML
