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

#include <rmm/rmm.h>

#include "../../../ml-prims/src/utils.h"

#include "../cuML.hpp"

namespace ML {

/**
 * @brief Implemententation of ML::deviceAllocator using the RAPIDS Memory Manager (RMM) for allocations.
 *
 * rmmAllocatorAdapter does not initialize RMM. If RMM is not initialized on construction of rmmAllocatorAdapter
 * allocations fall back to cudaMalloc.
 */
class rmmAllocatorAdapter : public ML::deviceAllocator {
public:
    rmmAllocatorAdapter()
        : _rmmInitialized( rmmIsInitialized( NULL ) )
    {
        //@todo: Log warning if RMM is not initialized. Blocked by https://github.com/rapidsai/cuml/issues/229
    }

    /**
     * @brief asynchronosly allocate n bytes that can be used after all work in stream sheduled prior to this call
     *        has completetd.
     *
     * @param[in] n         size of the allocation in bytes
     * @param[in] stream    the stream to use for the asynchronous allocations
     * @returns             a pointer to n byte of device memory
     */
    virtual void* allocate( std::size_t n, cudaStream_t stream )
    {
        void* ptr = 0;
        if (!_rmmInitialized)
        {
            CUDA_CHECK( cudaMalloc( &ptr, n ) );
        }
        else
        {
            rmmError_t rmmStatus = RMM_ALLOC(&ptr, n, stream);
            if ( RMM_SUCCESS != rmmStatus || 0 == ptr )
            {
                std::ostringstream msg;
                msg<<"RMM allocation of "<<n<<" byte failed: "<<rmmGetErrorString(rmmStatus)<<std::endl;;
                throw MLCommon::Exception(msg.str());
            }
        }
        return ptr;
    }

    /**
     * @brief asynchronosly free an allocation of n bytes that can be reused after all work in stream scheduled prior to this
     *        call has completed.
     *
     * @param[in] p         pointer to n bytes of memory to be deallocated
     * @param[in] n         size of the allocation to release in bytes
     * @param[in] stream    the stream to use for the asynchronous free
     */
    virtual void deallocate( void* p, std::size_t, cudaStream_t stream )
    {
        if (!_rmmInitialized)
        {
            cudaError_t status = cudaFree(p);
            if ( cudaSuccess != status )
            {
                //@todo: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
                // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
            }
        }
        else
        {
            rmmError_t rmmStatus = RMM_FREE(p, stream);
            if ( RMM_SUCCESS != rmmStatus )
            {
                //@todo: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
                // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
            }
        }
    }

    virtual ~rmmAllocatorAdapter() {}

private:
    const bool _rmmInitialized;
};

} // end namespace ML
