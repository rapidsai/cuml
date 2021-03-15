/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <string>
#include <vector>

namespace cuml{
namespace genetic{
  
/** fitness metric types */
enum class metric_t : uint32_t {
  /** mean absolute error (regression-only) */
  mae,
  /** mean squared error (regression-only) */
  mse,
  /** root mean squared error (regression-only) */
  rmse,
  /** pearson product-moment coefficient (regression and transformation) */
  pearson,
  /** spearman's rank-order coefficient (regression and transformation) */
  spearman,
  /** binary cross-entropy loss (classification-only) */
  logloss,
};  // enum class metric_t

} // namespace genetic
} // namespace cuml