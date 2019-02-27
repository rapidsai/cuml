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
 * RAII object owning a contigous typed device buffer. The passed in allocator supports asynchronus allocation and
 * deallocation so this can be used for temporary memory 
 * @code{.cpp}
 * template<typename T>
 * void foo( cumlHandle* handle, .., cudaStream_t stream )
 * {
 *     ...
 *     device_buffer<T> temp( handle->getDeviceAllocator(), 0 )
 *     
 *     temp.resize(n, stream);
 *     kernelA<<<grid,block,0,stream>>>(...,temp.data(),...);
 *     kernelB<<<grid,block,0,stream>>>(...,temp.data(),...);
 *     temp.release(stream);
 * }
 * @endcode
 */
template<typename T>
class device_buffer
{
public:
    typedef std::size_t         size_type;
    typedef T                   value_type;
    typedef value_type*         iterator;
    typedef const value_type*   const_iterator;
    typedef T&                  reference;
    typedef const T&            const_reference;

    device_buffer() = delete;

    device_buffer(const device_buffer& other) = delete;

    device_buffer& operator=(const device_buffer& other) = delete;

    device_buffer(std::shared_ptr<deviceAllocator> allocator, size_type n = 0)
        : _allocator(allocator), _size(n), _capacity(n), _data(nullptr)
    {
        if ( n > 0 )
        {
            _data = static_cast<value_type*>(_allocator->allocate( _capacity*sizeof(value_type), 0 ));
            CUDA_CHECK( cudaStreamSynchronize( 0 ) );
        }
    }

    ~device_buffer()
    {
        if ( nullptr != _data ) 
        {
            _allocator->deallocate( _data, _capacity*sizeof(value_type), 0 );
        }
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
    
    void resize(const size_type new_size, cudaStream_t stream )
    {
        if ( _capacity < new_size )
        {
            value_type* new_data = static_cast<value_type*>(_allocator->allocate( new_size*sizeof(value_type), stream ));
            if ( _size > 0 ) {
                CUDA_CHECK( cudaMemcpyAsync( new_data, _data, _size*sizeof(value_type), cudaMemcpyDeviceToDevice, stream ) );
            }
            if ( nullptr != _data ) {
                _allocator->deallocate( _data, _capacity*sizeof(value_type), stream );
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
        if ( nullptr != _data ) {
            _allocator->deallocate( _data, _capacity*sizeof(value_type), stream );
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
    
    std::shared_ptr<deviceAllocator> getAllocator() const
    {
        return _allocator;
    }

private:
    std::shared_ptr<deviceAllocator>    _allocator;
    size_type                           _size;
    size_type                           _capacity;
    value_type*                         _data;
};

} // end namespace ML
