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

#include <stdlib.h>
#include <vector>
#include <gmm/likelihood/handle.h>

namespace gmm {

template <typename T>
struct GMMHandle {
        llhdHandle_t<T> llhd_handle;
        T **dX_batches=NULL, **dmu_batches=NULL, **dsigma_batches=NULL,
        **dDiff_batches=NULL;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;
        T* dProbNorm;
        int lddprobnorm;
};

template <typename T>
struct GMM {
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;

        int lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs;

        T reg_covar, *cur_llhd;

        GMMHandle<T> handle;
};

}
