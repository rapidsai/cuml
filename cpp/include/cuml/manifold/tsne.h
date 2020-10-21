/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>

namespace ML {

/**
 * @brief Dimensionality reduction via TSNE using either Barnes Hut O(NlogN)
 *       or brute force O(N^2).
 *
 * @param[in]  handle              The GPU handle.
 * @param[in]  X                   The dataset you want to apply TSNE on.
 * @param[out] Y                   The final embedding.
 * @param[in]  n                   Number of rows in data X.
 * @param[in]  p                   Number of columns in data X.
 * @param[in]  dim                 Number of output dimensions for embeddings Y.
 * @param[in]  n_neighbors         Number of nearest neighbors used.
 * @param[in]  theta               Float between 0 and 1. Tradeoff for speed (0)
 *                                 vs accuracy (1) for Barnes Hut only.
 * @param[in]  epssq               A tiny jitter to promote numerical stability.
 * @param[in]  perplexity          How many nearest neighbors are used during
 *                                 construction of Pij.
 * @param[in]  perplexity_max_iter Number of iterations used to construct Pij.
 * @param[in]  perplexity_tol      The small tolerance used for Pij to ensure
 *                                 numerical stability.
 * @param[in]  early_exaggeration  How much early pressure you want the clusters
 *                                 in TSNE to spread out more.
 * @param[in] exaggeration_iter    How many iterations you want the early
 *                                 pressure to run for.
 * @param[in] min_gain             Rounds up small gradient updates.
 * @param[in] pre_learning_rate    The learning rate during exaggeration phase.
 * @param[in] post_learning_rate   The learning rate after exaggeration phase.
 * @param[in] max_iter             The maximum number of iterations TSNE should
 *                                 run for.
 * @param[in] min_grad_norm        The smallest gradient norm TSNE should
 *                                 terminate on.
 * @param[in] pre_momentum         The momentum used during the exaggeration
 *                                 phase.
 * @param[in] post_momentum        The momentum used after the exaggeration
 *                                 phase.
 * @param[in] random_state         Set this to -1 for pure random intializations
 *                                 or >= 0 for reproducible outputs.
 * @param[in] verbosity            verbosity level for logging messages during
 *                                 execution
 * @param[in] initialize_embeddings Whether to overwrite the current Y vector
 *                                 with random noise.
 * @param[in] barnes_hut           Whether to use the fast Barnes Hut or use the
 *                                 slower exact version.
 *
 * The CUDA implementation is derived from the excellent CannyLabs open source
 * implementation here: https://github.com/CannyLab/tsne-cuda/. The CannyLabs
 * code is licensed according to the conditions in
 * cuml/cpp/src/tsne/cannylabs_tsne_license.txt. A full description of their
 * approach is available in their article t-SNE-CUDA: GPU-Accelerated t-SNE and
 * its Applications to Modern Data (https://arxiv.org/abs/1807.11824).
 */
void TSNE_fit(const raft::handle_t &handle, const float *X, float *Y,
              const int n, const int p, const int dim = 2,
              int n_neighbors = 1023, const float theta = 0.5f,
              const float epssq = 0.0025, float perplexity = 50.0f,
              const int perplexity_max_iter = 100,
              const float perplexity_tol = 1e-5,
              const float early_exaggeration = 12.0f,
              const int exaggeration_iter = 250, const float min_gain = 0.01f,
              const float pre_learning_rate = 200.0f,
              const float post_learning_rate = 500.0f,
              const int max_iter = 1000, const float min_grad_norm = 1e-7,
              const float pre_momentum = 0.5, const float post_momentum = 0.8,
              const long long random_state = -1,
              int verbosity = CUML_LEVEL_INFO,
              const bool initialize_embeddings = true, bool barnes_hut = true);

}  // namespace ML
