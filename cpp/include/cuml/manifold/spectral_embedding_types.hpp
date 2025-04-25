/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cstdint>

namespace ML {

struct spectral_embedding_config {
  /** The number of components to reduce the data to. */
  int n_components;
  /** The number of neighbors to use for the nearest neighbors graph. */
  int n_neighbors;
  /** Whether to normalize the Laplacian matrix. */
  bool norm_laplacian;
  /** Whether to drop the first eigenvector. */
  bool drop_first;
  /** random seed */
  uint64_t seed;
};

}  // namespace ML
