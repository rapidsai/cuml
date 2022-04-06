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
#include <variant>

namespace ML {
namespace Explainer {

template <typename T>
class TreePathInfo;

using TreePathHandle = std::variant<std::shared_ptr<TreePathInfo <float >>,std::shared_ptr<TreePathInfo <double>>>;

TreePathHandle extract_path_info(ModelHandle model);

void gpu_treeshap(TreePathHandle path_info,
                  const float* data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  float* out_preds);

void gpu_treeshap(TreePathHandle path_info,
                  const double* data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  double * out_preds);

}  // namespace Explainer
}  // namespace ML