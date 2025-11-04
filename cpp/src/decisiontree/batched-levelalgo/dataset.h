/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
