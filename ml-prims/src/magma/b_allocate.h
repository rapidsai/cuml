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

#include <stdio.h>
#include <stdlib.h>

#include <magma_v2.h>
#include "cuda_utils.h"

namespace MLCommon {

template <typename T>
void create_pointer_array(T **&dA_array, magma_int_t n, magma_int_t batchCount){
        T **A_array;

        A_array = (T **)malloc(sizeof(T*) * batchCount);
        for(int i = 0; i < batchCount; i++) {
                allocate(A_array[i], n);
        }

        allocate(dA_array, batchCount);
        updateDevice(dA_array, A_array, batchCount);
        free(A_array);
}


template <typename T>
void allocate_pointer_array(T **&dA_array, magma_int_t n, magma_int_t batchCount){
        T **A_array;

        A_array = (T **)malloc(sizeof(T*) * batchCount);

        for(int i = 0; i < batchCount; i++) {
                allocate(A_array[i], n);
        }

        allocate(dA_array, batchCount);
        updateDevice(dA_array, A_array, batchCount);
        free(A_array);
}

}
