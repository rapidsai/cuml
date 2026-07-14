/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <cstdint>

namespace ML {
namespace DT {

template <typename DataT, typename LabelT, typename IdxT>
struct Dataset {
  /** input dataset */
  const DataT* data;
  /** input labels */
  const LabelT* labels;
  /** optional input sample weights */
  const double* sample_weight;
  /** total rows in dataset */
  std::int64_t n_rows;
  /** total cols in dataset */
  std::int64_t n_cols;
  /** row stride in input data elements */
  std::int64_t row_stride;
  /** column stride in input data elements */
  std::int64_t col_stride;
  /** total sampled rows in dataset */
  IdxT n_sampled_rows;
  /** total sampled cols in dataset */
  IdxT n_sampled_cols;
  /** indices of sampled rows */
  IdxT* row_ids;
  /** Number of classes or regression outputs*/
  IdxT num_outputs;

  HDI DataT value(IdxT row, IdxT col) const
  {
    return data[static_cast<std::int64_t>(row) * row_stride +
                static_cast<std::int64_t>(col) * col_stride];
  }
};

}  // namespace DT
}  // namespace ML
