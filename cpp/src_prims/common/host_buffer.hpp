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
#include "cuml_allocator.hpp"

namespace MLCommon {

/**
 * RAII object owning a contigous typed host buffer. The passed in allocator supports asynchronus allocation and
 * deallocation so this can be used for temporary memory 
 * @code{.cpp}
 * template<typename T>
 * void foo( const cumlHandle_impl& h, const T* in_d , T* out_d, ..., cudaStream_t stream )
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
 * @todo: Add missing doxygen documentation
 */
template<typename T>
class host_buffer : public buffer_base<T,hostAllocator>
{
public:
  using size_type         = typename buffer_base<T,hostAllocator>::size_type;
  using value_type        = typename buffer_base<T,hostAllocator>::value_type;
  using iterator          = typename buffer_base<T,hostAllocator>::iterator;
  using const_iterator    = typename buffer_base<T,hostAllocator>::const_iterator;
  using reference         = typename buffer_base<T,hostAllocator>::reference;
  using const_reference   = typename buffer_base<T,hostAllocator>::const_reference;

  host_buffer() = delete;

  host_buffer(const host_buffer& other) = delete;

  host_buffer& operator=(const host_buffer& other) = delete;

  host_buffer(std::shared_ptr<hostAllocator> allocator, cudaStream_t stream, size_type n = 0)
      : buffer_base<T,hostAllocator>(allocator,stream,n)
  {}

  ~host_buffer() {}

  reference operator[]( size_type pos )
  {
      return _data[pos];
  }

  const_reference operator[]( size_type pos ) const
  {
    return _data[pos];
  }

private:
  using buffer_base<T,hostAllocator>::_data;
};

} // end namespace ML
