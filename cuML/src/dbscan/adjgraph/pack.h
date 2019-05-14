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

namespace Dbscan {
namespace AdjGraph {

template <typename Type>
struct Pack {
    /**
     * vertex degree array
     * Last position is the sum of all elements in this array (excluding it)
     * Hence, its length is one more than the number of poTypes
     */
    int *vd;
    /** the adjacency matrix */
    bool *adj;     
    /** the adjacency graph */
    Type *adj_graph;

    Type adjnnz;

    /** exculusive scan generated from vd */ 
    Type *ex_scan;
    /** array to store whether a vertex is core poType or not */
    bool *core_pts;
    /** number of poTypes in the dataset */
    Type N;
    /** Minpts for classifying core pts */
    Type minPts;
};

} // namespace AdjGraph
} // namespace Dbscan
