/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/preprocessing/spectral/spectral_embedding.hpp>

namespace ML::SpectralEmbedding {

auto spectral_embedding_cuvs(raft::resources const& handle,
                             cuvs::preprocessing::spectral_embedding::params config,
                             raft::device_matrix_view<float, int, raft::row_major> dataset,
                             raft::device_matrix_view<float, int, raft::col_major> embedding)
  -> int;

}  // namespace ML::SpectralEmbedding
