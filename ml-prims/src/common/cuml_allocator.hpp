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

#include <cuda_runtime.h>
#include "utils.h"

namespace MLCommon {

/**
 * @brief Interface for a asynchronous device allocator.
 *
 * A implementation of this interface can make the following assumptions
 * - It does not need to be but it can allow asynchronous allocate and deallocate.
 * - Allocations may be always on the device that was specified on construction.
 */
class deviceAllocator {
public:
    /**
     * @brief Asynchronously allocates device memory.
     * 
     * An implementation of this need to return a allocation of n bytes properly align bytes
     * on the configured device. The allocation can optionally be asynchronous in the sense
     * that it is only save to use after all work submitted to the passed in stream prior to 
     * the call to allocate has completed. If the allocation is used before, e.g. in another 
     * stream the behaviour may be undefined.
     * @todo: Add alignment requirments.
     * 
     * @param[in] n         number of bytes to allocate
     * @param[in] stream    stream to issue the possible asynchronous allocation in
     * @returns a pointer to a n byte properly aligned device buffer on the configured device.
     */
    virtual void* allocate( std::size_t n, cudaStream_t stream ) = 0;
    /**
     * @brief Asynchronously deallocates device memory
     * 
     * An implementation of this need to ensure that the allocation that the passed in pointer
     * points to remains usable until all work sheduled in stream prior to the call to 
     * deallocate has completed.
     *
     * @param[in|out] p     pointer to the buffer to deallocte
     * @param[in] n         size of the buffer to deallocte in bytes
     * @param[in] stream    stream in which the allocation might be still in use
     */
    virtual void deallocate( void* p, std::size_t n, cudaStream_t stream ) = 0;

    virtual ~deviceAllocator() {}
};

/**
 * @brief Interface for a asynchronous host allocations.
 *
 * A implementation of this interface can make the following assumptions
 * - It does not need to be but it can allow asynchronous allocate and deallocate.
 * - Allocations don't need to be zero copy accessible form a device.
 */
class hostAllocator {
public:
    /**
     * @brief Asynchronously allocates host memory.
     * 
     * An implementation of this need to return a allocation of n bytes properly align bytes
     * on the host. The allocation can optionally be asynchronous in the sense
     * that it is only save to use after all work submitted to the passed in stream prior to 
     * the call to allocate has completed. If the allocation is used before, e.g. in another 
     * stream the behaviour may be undefined.
     * @todo: Add alignment requirments.
     * 
     * @param[in] n         number of bytes to allocate
     * @param[in] stream    stream to issue the possible asynchronous allocation in
     * @returns a pointer to a n byte properly aligned host buffer.
     */
    virtual void* allocate( std::size_t n, cudaStream_t stream ) = 0;
    /**
     * @brief Asynchronously deallocates host memory
     * 
     * An implementation of this need to ensure that the allocation that the passed in pointer
     * points to remains usable until all work sheduled in stream prior to the call to 
     * deallocate has completed.
     *
     * @param[in|out] p     pointer to the buffer to deallocte
     * @param[in] n         size of the buffer to deallocte in bytes
     * @param[in] stream    stream in which the allocation might be still in use
     */
    virtual void deallocate( void* p, std::size_t n, cudaStream_t stream ) = 0;

    virtual ~hostAllocator() {}
};


/** Default cudaMalloc/cudaFree based device allocator */
class defaultDeviceAllocator : public deviceAllocator {
public:
    virtual void* allocate( std::size_t n, cudaStream_t ) {
        void* ptr = 0;
        CUDA_CHECK( cudaMalloc( &ptr, n ) );
        return ptr;
    }
    virtual void deallocate( void* p, std::size_t, cudaStream_t ) {
        cudaError_t status = cudaFree( p);
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
    }

    virtual ~defaultDeviceAllocator() {}
};


/** Default cudaMallocHost/cudaFreeHost based host allocator */
class defaultHostAllocator : public hostAllocator {
public:
    virtual void* allocate( std::size_t n, cudaStream_t ) {
        void* ptr = 0;
        CUDA_CHECK( cudaMallocHost( &ptr, n ) );
        return ptr;
    }
    virtual void deallocate( void* p, std::size_t, cudaStream_t ) {
        cudaError_t status = cudaFreeHost( p);
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
    }

    virtual ~defaultHostAllocator() {}
};

}; // end namespace MLCommon
