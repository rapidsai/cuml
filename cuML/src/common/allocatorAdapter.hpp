/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
 * void foo( const cumlHandle_impl& h, ... , cudaStream_t stream )
 * {
 *     auto execution_policy = ML::exec_policy(h.getDeviceAllocator(),stream);
 *     thrust::for_each(execution_policy->on(stream), ... );
 * }
 * @endcode
 */
class thrustAllocatorAdapter
{
public:
    using value_type = char;

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

namespace 
{
    thrustAllocatorAdapter _decltypeHelper{0,0};
}

/**
 * @brief Returns a unique_ptr to a Thrust CUDA execution policy that uses the
 * passed in allocator for temporary memory allocation.
 *
 * @Param allocator The allocator to use
 * @Param stream    The stream that the allocator will use
 *
 * @Returns A Thrust execution policy that will use allocator for temporary memory
 * allocation.
 */
inline auto exec_policy(std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) -> std::unique_ptr<decltype(thrust::cuda::par(_decltypeHelper)),std::function<void(decltype(thrust::cuda::par(_decltypeHelper))*)> >
{
    thrustAllocatorAdapter * alloc{nullptr};

    alloc = new thrustAllocatorAdapter(allocator, stream);

    using T = decltype(thrust::cuda::par(*alloc));

    auto deleter = [alloc](T* pointer) {
        delete alloc;
        delete pointer;
    };
    
    std::unique_ptr<T, decltype(deleter)> policy{new T(*alloc), deleter};
    return policy;
}

} // end namespace ML
