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
#include "algo1.h"
#include "algo2.h"
#include "algo3.h"
#include "algo4.h"
#include <cutlass/gemm/dispatch.h>
#include "runner.h"
#include "pack.h"


namespace Dbscan {
namespace EpsNeigh {

using namespace cutlass::gemm;

template <typename Type>
void run(char* adj, Type* x, Type eps, int N, int D, cudaStream_t stream,
         int algo) {
    Pack<Type> data = {adj, x, eps, N, D};
    switch(algo) {
    case 0:
        Naive::launcher<Type>(data, stream);
        break;
    case 100:
        Algo1::launcher<Type, Algo1::P256>(data, stream);
        break;
    case 101:
        Algo1::launcher<Type, Algo1::P128>(data, stream);
        break;
    case 200:
        Algo2::launcher<Type, Algo2::P64x64<Type> >(data, stream);
        break;
    case 201:
        Algo2::launcher<Type, Algo2::P128x32<Type> >(data, stream);
        break;
    case 202:
        Algo2::launcher<Type, Algo2::P32x128<Type> >(data, stream);
        break;
    case 300:
        Algo3::launcher<Type, Algo3::P64x64<Type> >(data, stream);
        break;
    case 400:
        Algo4::launcher<Type, tiling_strategy::Small>(data, stream);
        break;
    case 401:
        Algo4::launcher<Type, tiling_strategy::Medium>(data, stream);
        break;
    case 402:
        Algo4::launcher<Type, tiling_strategy::Large>(data, stream);
        break;
    case 403:
        Algo4::launcher<Type, tiling_strategy::Tall>(data, stream);
        break;
    case 404:
        Algo4::launcher<Type, tiling_strategy::Wide>(data, stream);
        break;
    case 405:
        Algo4::launcher<Type, tiling_strategy::Huge>(data, stream);
        break;
    default:
        ASSERT(false, "Incorrect algo passed! '%d'", algo);
    }
}

} // namespace EpsNeigh
} // namespace Dbscan
