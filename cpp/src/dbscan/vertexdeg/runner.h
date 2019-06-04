/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "naive.h"
#include "pack.h"
#include "algo.h"
#include <common/cumlHandle.hpp>

namespace Dbscan {
namespace VertexDeg {


template <typename Type_f, typename Index_ = int>
void run(const ML::cumlHandle_impl& handle, bool* adj, int* vd, Type_f* x, Type_f eps,
         Index_ N, Index_ D,
         int algo, Index_ startVertexId, Index_ batchSize, cudaStream_t stream) {
    Pack<Type_f, Index_> data = {vd, adj, x, eps, N, D};
    switch(algo) {
    case 0:
        Naive::launcher<Type_f, Index_>(data, startVertexId, batchSize, stream);
    	break;
    case 1:
        Algo::launcher<Type_f, Index_>(handle, data, startVertexId, batchSize, stream);
    	break;
    default:
        ASSERT(false, "Incorrect algo passed! '%d'", algo);
    }
}

} // namespace VertexDeg
} // namespace Dbscan
