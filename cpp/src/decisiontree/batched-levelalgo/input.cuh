/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
namespace DecisionTree {

template <typename DataT, typename LabelT, typename IdxT>
struct Input {
  /** input dataset (assumed to be col-major) */
  const DataT* data;
  /** input labels */
  const LabelT* labels;
  /** total rows in dataset */
  IdxT M;
  /** total cols in dataset */
  IdxT N;
  /** total sampled rows in dataset */
  IdxT nSampledRows;
  /** total sampled cols in dataset */
  IdxT nSampledCols;
  /** indices of sampled rows */
  IdxT* rowids;
  /** number of classes (useful only in classification) */
  IdxT nclasses;
  /** quantiles/histogram computed on the dataset (col-major) */
  const DataT* quantiles;
};

}  // namespace DecisionTree
}  // namespace ML
