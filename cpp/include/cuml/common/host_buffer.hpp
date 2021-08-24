/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/mr/device/allocator.hpp>
#include <raft/mr/host/buffer.hpp>

namespace MLCommon {

/**
 * RAII object owning a contigous typed host buffer. The passed in allocator supports asynchronus
 * allocation and deallocation so this can be used for temporary memory
 * @code{.cpp}
 * template<typename T>
 * void foo( const raft::handle_t& h, const T* in_d , T* out_d, ..., cudaStream_t stream )
 * {
 *     ...
 *     host_buffer<T> temp( handle->get_host_allocator(), stream, 0 )
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

template <typename T>
using host_buffer = raft::mr::host::buffer<T>;

}  // namespace MLCommon
