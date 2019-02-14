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
    
class rmmAllocatorAdapter : public ML::deviceAllocator {
public:
    rmmAllocatorAdapter()
        : _rmmInitialized( rmmIsInitialized( NULL ) )
    {
        //@todo: Log warning if RMM is not initialized. Blocked by https://github.com/rapidsai/cuml/issues/229
    }

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

private:
    const bool _rmmInitialized;
};

} // end namespace ML
