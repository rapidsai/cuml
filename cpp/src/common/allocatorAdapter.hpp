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

#include <limits>

#include <thrust/system/cuda/execution_policy.h>

#include "../../src_prims/utils.h"

#include "../cuML.hpp"

namespace ML {

template<typename T>
class stdAllocatorAdapter
{
public:
    using size_type         = std::size_t;
    using value_type        = T;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
    using difference_type   = std::ptrdiff_t;

    template<typename U>
    struct rebind
    {
        typedef stdAllocatorAdapter<U> other;
    };

    stdAllocatorAdapter() = delete;

    stdAllocatorAdapter(const stdAllocatorAdapter& other) = default;

    template<typename U>
    stdAllocatorAdapter(stdAllocatorAdapter<U> const& other)
        : _allocator(other._allocator), _stream(other._stream)
    {}

    stdAllocatorAdapter& operator=(const stdAllocatorAdapter& other) = default;

    stdAllocatorAdapter(std::shared_ptr<hostAllocator> allocator, cudaStream_t stream)
        : _allocator(allocator), _stream(stream)
    {}

    ~stdAllocatorAdapter () {}

    inline pointer address(reference ref) const
    {
        return &ref;
    }
    inline const_pointer address(const_reference ref) const
    {
        return &ref;
    }

    pointer allocate(size_type size, typename std::allocator<void>::const_pointer = 0)
    {
        return static_cast<pointer>(_allocator->allocate( size, _stream ));
    }
    void deallocate(pointer ptr, size_type size) {
        _allocator->deallocate(ptr, size, _stream);
    }

    inline size_type max_size() const
    {
        return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }

    void construct(pointer ptr, const value_type& t) const
    {
        new(ptr) value_type(t);
    }
    void destroy(pointer ptr) const
    {
        ptr->~value_type();
    }

    bool operator==(const stdAllocatorAdapter&) const
    {
        return true;
    }
    bool operator!=(const stdAllocatorAdapter& other) const
    {
        return !operator==(other);
    }

private:
    std::shared_ptr<hostAllocator>  _allocator;
    cudaStream_t                    _stream = 0;
};

/**
 * @todo: Complete doxygen documentation
 * @code{.cpp}
 * void foo( const cumlHandle_impl& h, ... , cudaStream_t stream )
 * {
 *     auto execution_policy = ML::thrust_exec_policy(h.getDeviceAllocator(),stream);
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
 * @param[in] allocator The allocator to use
 * @param[in] stream    The stream that the allocator will use
 *
 * @returns A Thrust execution policy that will use allocator for temporary memory
 * allocation.
 */
inline auto thrust_exec_policy(std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream) -> std::unique_ptr<decltype(thrust::cuda::par(_decltypeHelper)),std::function<void(decltype(thrust::cuda::par(_decltypeHelper))*)> >
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
