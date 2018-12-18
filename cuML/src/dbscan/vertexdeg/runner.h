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
#include "naive.h"
#include "pack.h"
#include "algo.h"


namespace Dbscan {
namespace VertexDeg {


template <typename Type>
void run(bool* adj, int* vd, Type* x, Type* dots, Type eps, int N, int D,
         cudaStream_t stream, int algo, int startVertexId, int batchSize) {
         Pack<Type> data = {vd, adj, x, eps, N, D, dots};

    switch(algo) {

    case 0:
    	Naive::launcher(data, stream, startVertexId, batchSize);
    	break;

    case 1:
    	Algo::launcher<Type>(data, stream, startVertexId, batchSize);
    	break;

    default:
        ASSERT(false, "Incorrect algo passed! '%d'", algo);
    }

}

} // namespace VertexDeg
} // namespace Dbscan
