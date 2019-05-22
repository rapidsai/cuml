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

#include "algo.h"
#include "pack.h"
#include "naive.h"
#include <common/cumlHandle.hpp>

namespace Dbscan {
namespace AdjGraph {

template <typename Type>
void run(const ML::cumlHandle_impl& handle, bool* adj, int* vd, Type* adj_graph, Type adjnnz, Type* ex_scan, Type N,
         Type minpts, bool* core_pts, int algo, Type batchSize, cudaStream_t stream) {
    Pack<Type> data = {vd, adj, adj_graph, adjnnz, ex_scan, core_pts, N, minpts};
    switch(algo) {
    case 0:
        Naive::launcher<Type>(handle, data, batchSize, stream);
        break;
    case 1:
        Algo::launcher<Type>(handle, data, batchSize, stream);
        break;
    default:
        ASSERT(false, "Incorrect algo passed! '%d'", algo);
    }
}


} // namespace AdjGraph
} // namespace Dbscan
