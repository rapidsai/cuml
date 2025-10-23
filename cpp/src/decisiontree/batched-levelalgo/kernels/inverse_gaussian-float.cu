/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../bins.cuh"
#include "../objectives.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using _DataT      = float;
using _LabelT     = float;
using _IdxT       = int;
using _ObjectiveT = InverseGaussianObjectiveFunction<_DataT, _LabelT, _IdxT>;
using _BinT       = AggregateBin;
using _DatasetT   = Dataset<_DataT, _LabelT, _IdxT>;
using _NodeT      = SparseTreeNode<_DataT, _LabelT, _IdxT>;
}  // namespace DT
}  // namespace ML

#include "builder_kernels_impl.cuh"
