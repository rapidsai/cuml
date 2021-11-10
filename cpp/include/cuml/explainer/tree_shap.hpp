/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/ensemble/treelite_defs.hpp>
#include <cstddef>

namespace ML {
namespace Explainer {

typedef void* ExtractedPathHandle;

void extract_paths(ModelHandle model, ExtractedPathHandle* extracted_paths);
void gpu_treeshap(ExtractedPathHandle extracted_paths, const float* data,
                  std::size_t n_rows, std::size_t n_cols, float* out_preds);
void free_extracted_paths(ExtractedPathHandle extracted_paths);

}  // namespace Explainer
}  // namespace ML
