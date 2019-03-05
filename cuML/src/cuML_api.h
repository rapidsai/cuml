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

#include <stddef.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    void* ptr;
} cumlHandle_t;

enum cumlError_t { CUML_SUCCESS, CUML_ERROR_UNKOWN };

typedef cudaError_t (*cuml_allocate)(void** p,size_t n, cudaStream_t stream);
typedef cudaError_t (*cuml_deallocate)(void* p, size_t n, cudaStream_t stream);

const char* cumlGetErrorString ( cumlError_t error );

cumlError_t cumlCreate( cumlHandle_t* handle );

cumlError_t cumlSetStream( cumlHandle_t handle, cudaStream_t stream );
cumlError_t cumlGetStream( cumlHandle_t handle, cudaStream_t* stream );

/**
 * @code{.c}
 * cudaError_t device_allocate(void** p,size_t n, cudaStream_t)
 * {
 *     return cudaMalloc(p,n);
 * }
 * 
 * cudaError_t device_deallocate(void* p, size_t, cudaStream_t)
 * {
 *     return cudaFree(p);
 * }
 * 
 * void foo()
 * {
 *     cumlHandle_t cumlHandle;
 *     cumlCreate( &cumlHandle );

 *     cumlSetDeviceAllocator( cumlHandle, device_allocate, device_deallocate );

 *     cumlDestroy( cumlHandle );
 * }
 * @endcode
 */
cumlError_t cumlSetDeviceAllocator( cumlHandle_t handle, cuml_allocate allocate_fn, cuml_deallocate deallocate_fn );
/**
 * @code{.c}
 * cudaError_t host_allocate(void** p,size_t n, cudaStream_t)
 * {
 *     *p = malloc(n);
 *     return NULL != *p ? cudaSuccess : cudaErrorUnknown;
 * }
 * 
 * cudaError_t host_deallocate(void* p, size_t, cudaStream_t stream)
 * {
 *     free(p);
 *     return cudaSuccess;
 * }
 * 
 * void foo()
 * {
 *     cumlHandle_t cumlHandle;
 *     cumlCreate( &cumlHandle );

 *     cumlSetHostAllocator( cumlHandle, host_allocate, host_deallocate );

 *     cumlDestroy( cumlHandle );
 * }
 * @endcode
 */
cumlError_t cumlSetHostAllocator( cumlHandle_t handle, cuml_allocate allocate_fn, cuml_deallocate deallocate_fn );

cumlError_t cumlDestroy( cumlHandle_t handle );

#ifdef __cplusplus
}
#endif
