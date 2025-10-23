/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ML {
namespace DT {

template <typename DataT, typename IdxT>
struct Quantiles {
  /** quantiles computed for each feature of dataset in col-major */
  DataT* quantiles_array;
  /** The number of bins used for quantiles of each feature*/
  IdxT* n_bins_array;
};

}  // namespace DT
}  // namespace ML
