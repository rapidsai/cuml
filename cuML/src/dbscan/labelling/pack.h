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
namespace Label {

template <typename Type>
struct Pack {
    /**
     * vertex degree array
     * Last position is the sum of all elements in this array (excluding it)
     * Hence, its length is one more than the number of poTypes
     */
    Type *vd;
    /** the adjacency matrix */
    bool *adj;
    /** the adjacency graph */
    Type *adj_graph;
    /** exculusive scan generated from vd */
    Type *ex_scan;
    /** array to store whether a vertex is core poType or not */
    bool *core_pts;
    /** number of poTypes in the dataset */
    Type N;
    /** Minpts for classifying core pts */
    Type minPts;
    /** arra to store visited points */ 
    bool *visited;
    /** array to store the final cluster */
    Type *db_cluster;
    /** array to store visited points for GPU */ 
    bool *xa;
    /** array to store border points for GPU */
    bool *fa;
    /** bool variable for algo 2 */ 
    bool *m;
    /** array to store map index after sorting */
    Type *map_id;

    void resetArray(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(visited, false, sizeof(bool)*N, stream));
        CUDA_CHECK(cudaMemsetAsync(db_cluster, 0, sizeof(Type)*N, stream));
        CUDA_CHECK(cudaMemsetAsync(xa, false, sizeof(bool)*N, stream));
        CUDA_CHECK(cudaMemsetAsync(fa, false, sizeof(bool)*N, stream));
        CUDA_CHECK(cudaMemsetAsync(map_id, 0, sizeof(Type)*N, stream));
    }	
};

} // namespace Label
} // namespace Dbscan
