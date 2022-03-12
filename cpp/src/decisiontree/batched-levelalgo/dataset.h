/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

namespace ML {
namespace DT {

template <typename DataT, typename LabelT, typename IdxT>
struct Dataset {
  /** input dataset (assumed to be col-major) */
  const DataT* data;
  /** input labels */
  const LabelT* labels;
  /** total rows in dataset */
  IdxT M;
  /** total cols in dataset */
  IdxT N;
  /** total sampled rows in dataset */
  IdxT n_sampled_rows;
  /** total sampled cols in dataset */
  IdxT n_sampled_cols;
  /** indices of sampled rows */
  IdxT* row_ids;
  /** Number of classes or regression outputs*/
  IdxT num_outputs;
};

}  // namespace DT
}  // namespace ML
