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
 * RAII object owning a contigous typed host buffer. The passed in allocator supports asynchronus allocation and
 * deallocation so this can be used for temporary memory 
 * @code{.cpp}
 * template<typename T>
 * void foo( cumlHandle* handle, const T* in_d , T* out_d, ..., cudaStream_t stream )
 * {
 *     ...
 *     host_buffer<T> temp( handle->getHostAllocator(), stream, 0 )
 *     
 *     temp.resize(n, stream);
 *     cudaMemcpyAsync( temp.data(), in_d, temp.size()*sizeof(T), cudaMemcpyDeviceToHost );
 *     ...
 *     cudaMemcpyAsync( out_d, temp.data(), temp.size()*sizeof(T), cudaMemcpyHostToDevice );
 *     temp.release(stream);
 * }
 * @endcode
 */
template<typename T>
class host_buffer
{
public:
    using size_type         = std::size_t;
    using value_type        = T;
    using iterator          = value_type*;
    using const_iterator    = const value_type*;
    using reference         = T&;
    using const_reference   = const T&;

    host_buffer() = delete;

    host_buffer(const host_buffer& other) = delete;

    host_buffer& operator=(const host_buffer& other) = delete;

    host_buffer(std::shared_ptr<hostAllocator> allocator, cudaStream_t stream, size_type n = 0)
        : _allocator(allocator), _size(n), _capacity(n), _data(nullptr), _stream(stream)
    {
        if ( _capacity > 0 )
        {
            _data = static_cast<value_type*>(_allocator->allocate( _capacity*sizeof(value_type), _stream ));
            CUDA_CHECK( cudaStreamSynchronize( _stream ) );
        }
    }

    ~host_buffer()
    {
        if ( nullptr != _data ) 
        {
            _allocator->deallocate( _data, _capacity*sizeof(value_type), _stream );
        }
    }

    reference operator[]( size_type pos )
    {
        return _data[pos];
    }

    const_reference operator[]( size_type pos ) const
    {
        return _data[pos];
    }

    value_type* data()
    {
        return _data;
    }

    const value_type* data() const
    {
        return _data;
    }
    
    size_type size() const
    {
        return _size;
    }
    
    void reserve( const size_type new_capacity, cudaStream_t stream )
    {
        _stream = stream;
        if ( new_capacity > _capacity )
        {
            value_type* new_data = static_cast<value_type*>(_allocator->allocate( new_capacity*sizeof(value_type), _stream ));
            if ( _size > 0 ) {
                CUDA_CHECK( cudaMemcpyAsync( new_data, _data, _size*sizeof(value_type), cudaMemcpyHostToHost, _stream ) );
            }
            if ( nullptr != _data ) {
                _allocator->deallocate( _data, _capacity*sizeof(value_type), _stream );
            }
            _data = new_data;
            _capacity = new_capacity;
        }
    }
    
    void resize(const size_type new_size, cudaStream_t stream )
    {
        _stream = stream;
        if ( _capacity < new_size )
        {
            value_type* new_data = static_cast<value_type*>(_allocator->allocate( new_size*sizeof(value_type), _stream ));
            if ( _size > 0 ) {
                CUDA_CHECK( cudaMemcpyAsync( new_data, _data, _size*sizeof(value_type), cudaMemcpyHostToHost, _stream ) );
            }
            if ( nullptr != _data ) {
                _allocator->deallocate( _data, _capacity*sizeof(value_type), _stream );
            }
            _data = new_data;
            _capacity = new_size;
        }
        _size = new_size;
    }
    
    void clear()
    {
        _size = 0;
    }
    
    void release( cudaStream_t stream )
    {
        _stream = stream;
        if ( nullptr != _data ) {
            _allocator->deallocate( _data, _capacity*sizeof(value_type), _stream );
        }
        _data = nullptr;
        _capacity = 0;
        _size = 0;
    }
    
    iterator begin()
    {
        return _data;
    }
    
    const_iterator begin() const
    {
        return _data;
    }
    
    iterator end()
    {
        return _data+_size;
    }
    
    const_iterator end() const
    {
        return _data+_size;
    }

    std::shared_ptr<hostAllocator> getAllocator() const
    {
        return _allocator;
    }

private:
    std::shared_ptr<hostAllocator>      _allocator;
    size_type                           _size;
    size_type                           _capacity;
    value_type*                         _data;
    cudaStream_t                        _stream;
};

} // end namespace ML
