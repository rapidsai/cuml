/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "../bins.cuh"
#include "../objectives.cuh"

#include <cuml/tree/flatnode.h>

namespace ML {
namespace DT {
using _DataT      = double;
using _LabelT     = double;
using _IdxT       = int;
using _ObjectiveT = GammaObjectiveFunction<_DataT, _LabelT, _IdxT>;
using _BinT       = AggregateBin;
using _DatasetT   = Dataset<_DataT, _LabelT, _IdxT>;
using _NodeT      = SparseTreeNode<_DataT, _LabelT, _IdxT>;
}  // namespace DT
}  // namespace ML

#include "builder_kernels_impl.cuh"
