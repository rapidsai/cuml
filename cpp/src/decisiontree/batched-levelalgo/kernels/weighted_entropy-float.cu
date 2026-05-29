/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../bins.cuh"
#include "../objectives.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using _DataT      = float;
using _LabelT     = int;
using _IdxT       = int;
using _ObjectiveT = WeightedEntropyObjectiveFunction<_DataT, _LabelT, _IdxT>;
using _BinT       = WeightedCountBin;
using _DatasetT   = Dataset<_DataT, _LabelT, _IdxT>;
using _NodeT      = SparseTreeNode<_DataT, _LabelT, _IdxT>;
}  // namespace DT
}  // namespace ML

// class_weight='balanced_subsample' needs the HasTreeClassWeight=true
// instantiation of launchComputeSplitKernel on the weighted-classifier path.
#define INSTANTIATE_TREE_CLASS_WEIGHT
#include "builder_kernels_impl.cuh"
