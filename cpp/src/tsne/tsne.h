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

#include "cuML.hpp"

#pragma once

namespace ML {

void TSNE_fit(const cumlHandle &handle, const float *X, float *Y, const int n,
              const int p, const int dim = 2, int n_neighbors = 1023,
              const float theta = 0.5f, const float epssq = 0.0025,
              float perplexity = 50.0f, const int perplexity_max_iter = 100,
              const float perplexity_tol = 1e-5,
              const float early_exaggeration = 12.0f,
              const int exaggeration_iter = 250, const float min_gain = 0.01f,
              const float pre_learning_rate = 200.0f,
              const float post_learning_rate = 500.0f,
              const int max_iter = 1000, const float min_grad_norm = 1e-7,
              const float pre_momentum = 0.5, const float post_momentum = 0.8,
              const long long random_state = -1, const bool verbose = true,
              const bool intialize_embeddings = true, bool barnes_hut = true);

}
