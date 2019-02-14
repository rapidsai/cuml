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

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    void* ptr;
} cumlHandle_t;

enum cumlError_t { CUML_SUCCESS, CUML_ERROR_UNKOWN };

const char* cumlGetErrorString ( cumlError_t error );

cumlError_t cumlCreate( cumlHandle_t* handle );

cumlError_t cumlSetStream( cumlHandle_t handle, cudaStream_t stream );

cumlError_t cumlDestroy( cumlHandle_t handle );

#ifdef __cplusplus
}
#endif
