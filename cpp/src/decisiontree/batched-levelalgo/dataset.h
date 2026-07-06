/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace ML {
namespace DT {

template <typename DataT, typename LabelT>
struct Dataset {
  /** input dataset (assumed to be col-major) */
  const DataT* data;
  /** input labels */
  const LabelT* labels;
  /** optional input sample weights */
  const double* sample_weight;
  /** total rows in dataset */
  std::int64_t M;
  /** total cols in dataset */
  std::int64_t N;
  /** total sampled rows in dataset */
  std::int64_t n_sampled_rows;
  /** total sampled cols in dataset */
  std::int64_t n_sampled_cols;
  /** indices of sampled rows */
  std::int64_t* row_ids;
  /** Number of classes or regression outputs*/
  std::int64_t num_outputs;
};

}  // namespace DT
}  // namespace ML
