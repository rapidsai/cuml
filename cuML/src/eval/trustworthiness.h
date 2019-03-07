/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "distance/distance.h"


namespace Evaluation {

    template<typename T>
    void trustworthiness(T *X, int m, int n, T *embedded, int n_components, int n_neighbors) {

        T *dist;
        MLCommon::allocate(dist, m*m);

        MLCommon::Distance::distance<value_t, value_t, value_t, OutputTile_t>
                (data.x, data.x+startVertexId*k,                    // x & y inputs
                     dist,
                 m, n, n,                                           // Cutlass block params
                 MLCommon::Distance::DistanceType::EucExpandedL2,   // distance metric type
                 (void*)workspace, workspaceSize,                           // workspace params
                 stream                                             // cuda stream
        );





    }


}
