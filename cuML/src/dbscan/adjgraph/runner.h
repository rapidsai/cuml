/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

namespace Dbscan {
namespace AdjGraph {

template <typename Type>
void run(bool* adj, Type* vd, Type* adj_graph, Type* ex_scan, Type N,
         Type minpts, bool* core_pts, cudaStream_t stream, int algo, int batchSize) {
    Pack<Type> data = {vd, adj, adj_graph, ex_scan, core_pts, N, minpts};
    switch(algo) {
    case 0:
        Naive::launcher<Type>(data, batchSize, stream);
        break;
    case 1:
        Algo::launcher<Type>(data, batchSize, stream);
        break;
    default:
        ASSERT(false, "Incorrect algo passed! '%d'", algo);
    }
}


} // namespace AdjGraph
} // namespace Dbscan
