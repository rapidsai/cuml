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

#include "magma/b_handles.h"

using namespace MLCommon;

namespace gmm {

template <typename T>
struct llhd_bilinearHandle_t {
        T* dT;
        T **dT_vec_array, **dT_mat_array;

        int lddt, batchCount, nCl, nObs;
};

template <typename T>
struct llhdHandle_t {

        T **dInvSigma_array, *dInvdet_array, **dX_batches, **dmu_batches, **dInvSigma_batches, **dDiff_batches, *dBil_batches, **dDiff_mat_array,
        **dDiff_vec_array;

        T* dInvSigma, *dDiff;
        size_t dDiff_size;
        int lddDiff;

        int nCl, nObs, lddx, lddsigma_full, batchCount;

        T *bilinearWs, *determinantWs, *inverseWs;

        llhd_bilinearHandle_t<T> bilinearHandle;
        determinantHandle_t<T> determinantHandle;
        inverseHandle_t<T> inverseHandle;
};

}
