/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
  int M;
  /** total cols in dataset */
  int N;
  /** total sampled rows in dataset */
  int n_sampled_rows;
  /** total sampled cols in dataset */
  int n_sampled_cols;
  /** indices of sampled rows */
  int* row_ids;
  /** Number of classes or regression outputs*/
  int num_outputs;
};

}  // namespace DT
}  // namespace ML
