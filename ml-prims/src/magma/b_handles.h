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

namespace MLCommon {
template <typename T>
struct bilinearHandle_t {
        T* dT;
        T **dT_array;

        int lddt, batchCount;
};

template <typename T>
struct determinantHandle_t {
        int* dipiv, *info_array;
        T* dA_cpy;

        int **dipiv_array;
        T **dA_cpy_array; // U and L are stored here after getrf

        int ldda, batchCount, n;
};

template <typename T>
struct inverseHandle_t {
        int* dipiv, *info_array;
        T* dA_cpy;

        int **dipiv_array;
        T **dA_cpy_array; // U and L are stored here after getrf

        int ldda, batchCount, n;
};
}
