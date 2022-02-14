/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>
#include <cuml/ensemble/treelite_defs.hpp>
#include <memory>

namespace ML {
namespace Explainer {

// An abstract class representing an opaque handle to path information
// extracted from a tree model. The implementation in tree_shap.cu will
// define an internal class that inherits from this abtract class.
class TreePathInfo {
 public:
  enum class ThresholdTypeEnum : std::uint8_t { kFloat, kDouble };
  virtual ThresholdTypeEnum GetThresholdType() const = 0;
};

std::unique_ptr<TreePathInfo> extract_path_info(ModelHandle model);
void gpu_treeshap(TreePathInfo* path_info,
                  const float* data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  float* out_preds);

}  // namespace Explainer
}  // namespace ML
