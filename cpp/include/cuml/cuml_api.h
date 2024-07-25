/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <stddef.h>

// Block inclusion of this header when compiling libcuml++.so. If this error is
// shown during compilation, there is an issue with how the `#include` have
// been set up. To debug the issue, run `./build.sh cppdocs` and open the page
// 'cpp/build/html/cuml__api_8h.html' in a browser. This will show which files
// directly and indirectly include this file. Only files ending in '*_api' or
// 'cumlHandle' should include this header.
#ifdef CUML_CPP_API
#error "This header is only for the C-API and should not be included from the C++ API."
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int cumlHandle_t;

typedef enum cumlError_t { CUML_SUCCESS, CUML_ERROR_UNKNOWN, CUML_INVALID_HANDLE } cumlError_t;

typedef cudaError_t (*cuml_allocate)(void** p, size_t n, cudaStream_t stream);
typedef cudaError_t (*cuml_deallocate)(void* p, size_t n, cudaStream_t stream);

/**
 * @brief Get a human readable error string for the passed in error code.
 *
 * @param[in] error the error code to decipher.
 * @return a string with a human readable error message.
 */
const char* cumlGetErrorString(cumlError_t error);

/**
 * @brief Creates a cumlHandle_t
 *
 * @param[inout] handle     pointer to the handle to create.
 * @param[in] stream        the stream to which cuML work should be ordered.
 * @return CUML_SUCCESS on success, @todo: add more error codes
 */
cumlError_t cumlCreate(cumlHandle_t* handle, cudaStream_t stream);

/**
 * @brief sets the stream to which all cuML work issued via the passed handle should be ordered.
 *
 * @param[inout] handle     handle to set the stream for.
 * @param[in] stream        the stream to which cuML work should be ordered.
 * @return CUML_SUCCESS on success, @todo: add more error codes
 */
cumlError_t cumlGetStream(cumlHandle_t handle, cudaStream_t* stream);

/**
 * @brief sets the allocator to use for all device allocations done in cuML.
 *
 * Example use:
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
 * @param[inout] handle     the cumlHandle_t to set the device allocator for.
 * @param[in] allocate_fn    function pointer to the allocate function to use for device
 allocations.
 * @param[in] deallocate_fn  function pointer to the deallocate function to use for device
 allocations.
 * @return CUML_SUCCESS on success, @todo: add more error codes
 */
cumlError_t cumlSetDeviceAllocator(cumlHandle_t handle,
                                   cuml_allocate allocate_fn,
                                   cuml_deallocate deallocate_fn);
/**
 * @brief sets the allocator to use for substantial host allocations done in cuML.
 *
 * Example use:
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
 * @param[inout] handle      the cumlHandle_t to set the host allocator for.
 * @param[in] allocate_fn    function pointer to the allocate function to use for host allocations.
 * @param[in] deallocate_fn  function pointer to the deallocate function to use for host
 allocations.
 * @return CUML_SUCCESS on success, @todo: add more error codes
 */
cumlError_t cumlSetHostAllocator(cumlHandle_t handle,
                                 cuml_allocate allocate_fn,
                                 cuml_deallocate deallocate_fn);

/**
 * @brief Release all resource internally managed by cumlHandle_t
 *
 * @param[inout] handle     the cumlHandle_t to destroy.
 * @return CUML_SUCCESS on success, @todo: add more error codes
 */
cumlError_t cumlDestroy(cumlHandle_t handle);

#ifdef __cplusplus
}
#endif
