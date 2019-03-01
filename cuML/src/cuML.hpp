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

#include <memory>

#include <cuda_runtime.h>

#include <common/cuml_allocator.hpp>

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

class cumlHandle_impl;

class cumlHandle {
public:
    cumlHandle();
    ~cumlHandle();
    void setStream( cudaStream_t stream );
    cudaStream_t getStream() const;
    void setDeviceAllocator( std::shared_ptr<deviceAllocator> allocator );
    std::shared_ptr<deviceAllocator> getDeviceAllocator() const;
    void setHostAllocator( std::shared_ptr<hostAllocator> allocator );
    std::shared_ptr<hostAllocator> getHostAllocator() const;
    const cumlHandle_impl& getImpl() const;
private:
    std::unique_ptr<cumlHandle_impl> _impl;
};

} // end namespace ML
