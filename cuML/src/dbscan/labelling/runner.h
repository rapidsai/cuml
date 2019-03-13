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

#include <cuda_utils.h>
#include <iostream>
#include <limits>
#include "naive.h"
#include "algo1.h"
#include "pack.h"
#include "algo2.h"
#include <cuML.hpp>

namespace Dbscan {
namespace Label {


template <typename Type>
void run(const ML::cumlHandle& handle, bool* adj, int* vd, Type* adj_graph, Type* ex_scan, Type N,
         Type minpts, bool* core_pts, bool* visited, Type *db_cluster, 
         bool *xa, bool *fa, bool *m, Type *map_id, 
         int algo, int startVertexId, int batchSize) {
    Pack<Type> data = {vd, adj, adj_graph, ex_scan, core_pts, N, minpts,
                       visited, db_cluster, xa, fa, m, map_id};
    switch(algo) {
    case 0:
        Naive::launcher<Type>(handle, data, startVertexId, batchSize);
        break;
    case 1:
        ASSERT(N == batchSize, "Label::Algo1 doesn't support batching!");
        Algo1::launcher<Type>(handle, data, startVertexId, batchSize);
        break;
    case 2:
        Algo2::launcher<Type>(handle, data, N, startVertexId, batchSize);
        break;
    default:
        ASSERT(false, "Incorrect algo passed! '%d'", algo);
    }
}

template <typename Type>
void final_relabel(const ML::cumlHandle& handle, bool* adj, int* vd, Type* adj_graph, Type* ex_scan, Type N,
         Type minpts, bool* core_pts, bool* visited, Type *db_cluster,
         bool *xa, bool *fa, bool *m, Type *map_id) {
    Pack<Type> data = {vd, adj, adj_graph, ex_scan, core_pts, N, minpts,
                       visited, db_cluster, xa, fa, m, map_id};
    Algo2::relabel<Type>(handle, data);
}

} // namespace Label
} // namespace Dbscan

