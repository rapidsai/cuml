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

#include <raft/mr/device/buffer.hpp>

namespace MLCommon {

/**
 * RAII object owning a contigous typed device buffer. The passed in allocator supports asynchronus allocation and
 * deallocation so this can be used for temporary memory 
 * @code{.cpp}
 * template<typename T>
 * void foo( const raft::handle_t& h, ..., cudaStream_t stream )
 * {
 *     ...
 *     device_buffer<T> temp( h.get_device_allocator(), stream, 0 )
 *     
 *     temp.resize(n, stream);
 *     kernelA<<<grid,block,0,stream>>>(...,temp.data(),...);
 *     kernelB<<<grid,block,0,stream>>>(...,temp.data(),...);
 *     temp.release(stream);
 * }
 * @endcode
 */
template <typename T>
using device_buffer = raft::mr::device::buffer<T>;

}  // namespace MLCommon
