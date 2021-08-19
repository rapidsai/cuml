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
namespace DT {

template <typename DataT, typename LabelT>
struct Input {
  /** input dataset (assumed to be col-major) */
  const DataT* data;
  /** input labels */
  const LabelT* labels;
  /** total rows in dataset */
  std::size_t M;
  /** total cols in dataset */
  std::size_t N;
  /** total sampled rows in dataset */
  std::size_t nSampledRows;
  /** total sampled cols in dataset */
  std::size_t nSampledCols;
  /** indices of sampled rows */
  std::size_t* rowids;
  /** Number of classes or regression outputs*/
  std::size_t numOutputs;
  /** quantiles/histogram computed on the dataset (col-major) */
  const DataT* quantiles;
};

}  // namespace DT
}  // namespace ML
