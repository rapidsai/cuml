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

#include "buffer_base.hpp"

#include "../utils.h"

#include "cuml_allocator.hpp"

namespace MLCommon {

/**
 * RAII object owning a contigous typed device buffer. The passed in allocator supports asynchronus allocation and
 * deallocation so this can be used for temporary memory 
 * @code{.cpp}
 * template<typename T>
 * void foo( const cumlHandle_impl& h, ..., cudaStream_t stream )
 * {
 *     ...
 *     device_buffer<T> temp( h.getDeviceAllocator(), stream, 0 )
 *     
 *     temp.resize(n, stream);
 *     kernelA<<<grid,block,0,stream>>>(...,temp.data(),...);
 *     kernelB<<<grid,block,0,stream>>>(...,temp.data(),...);
 *     temp.release(stream);
 * }
 * @endcode
 * @todo: Add missing doxygen documentation
 */
template<typename T>
class device_buffer : public buffer_base<T>
{
public:
    using size_type         = typename buffer_base<T>::size_type;
    using value_type        = typename buffer_base<T>::value_type;
    using iterator          = typename buffer_base<T>::iterator;
    using const_iterator    = typename buffer_base<T>::const_iterator;
    using reference         = typename buffer_base<T>::reference;
    using const_reference   = typename buffer_base<T>::const_reference;
    using buffer_base<T>::get_stream;
    using buffer_base<T>::set_stream;

    device_buffer() = delete;

    device_buffer(const device_buffer& other) = delete;

    device_buffer& operator=(const device_buffer& other) = delete;

    device_buffer(std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream, size_type n = 0)
        : buffer_base<T>(stream,n), _allocator(allocator)
    {
        if ( _size > 0 )
        {
            _data = static_cast<value_type*>(_allocator->allocate( _capacity*sizeof(value_type), get_stream() ));
            CUDA_CHECK( cudaStreamSynchronize( get_stream() ) );
        }
    }

    ~device_buffer()
    {
        if ( nullptr != _data ) 
        {
            _allocator->deallocate( _data, _capacity*sizeof(value_type), get_stream() );
        }
    }
    
    void reserve( const size_type new_capacity, cudaStream_t stream )
    {
        set_stream( stream );
        if ( new_capacity > _capacity )
        {
            value_type* new_data = static_cast<value_type*>(_allocator->allocate( new_capacity*sizeof(value_type), get_stream() ));
            if ( _size > 0 ) {
                CUDA_CHECK( cudaMemcpyAsync( new_data, _data, _size*sizeof(value_type), cudaMemcpyDeviceToDevice, get_stream() ) );
            }
            if ( nullptr != _data ) {
                _allocator->deallocate( _data, _capacity*sizeof(value_type), get_stream() );
            }
            _data = new_data;
            _capacity = new_capacity;
        }
    }
    
    void resize(const size_type new_size, cudaStream_t stream )
    {
        reserve( new_size, stream );
        _size = new_size;
    }
    
    void release( cudaStream_t stream )
    {
        set_stream( stream );
        if ( nullptr != _data ) {
            _allocator->deallocate( _data, _capacity*sizeof(value_type), get_stream() );
        }
        _data = nullptr;
        _capacity = 0;
        _size = 0;
    }
    
    std::shared_ptr<deviceAllocator> getAllocator() const
    {
        return _allocator;
    }

private:
    std::shared_ptr<deviceAllocator>    _allocator;
    using buffer_base<T>::_size;
    using buffer_base<T>::_capacity;
    using buffer_base<T>::_data;
};

} // end namespace ML
