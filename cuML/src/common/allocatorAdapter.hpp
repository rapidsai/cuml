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

#include "../../../ml-prims/src/utils.h"

#include "../cuML.hpp"

namespace ML {
    
    
/**
 * @code{.cpp}
 * void foo( cumlHandle* handle, .. )
 * {
 *     thrustAllocatorAdapter alloc( handle->getDeviceAllocator(), handle->getStream() );
 *     auto execution_policy = thrust::cuda::par(alloc).on(handle->getStream());
 *     thrust::for_each(execution_policy, ... );
 * }
 * @endcode
 */
class thrustAllocatorAdapter
{
public:
    typedef char value_type;

    thrustAllocatorAdapter() = delete;

    thrustAllocatorAdapter(std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream)
        : _allocator(allocator), _stream(stream)
    {}
    
    ~thrustAllocatorAdapter() {}
    
    char* allocate(const size_t size)
    {
        return static_cast<char*>(_allocator->allocate( size, _stream ));
    }

    void deallocate(char* ptr, const size_t size)
    {
        _allocator->deallocate( ptr, size, _stream );
    }

private:
    std::shared_ptr<deviceAllocator>    _allocator;
    cudaStream_t                        _stream = 0;
};

} // end namespace ML
