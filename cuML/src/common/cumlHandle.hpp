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

#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "../cuML.hpp"

namespace ML {

/**
 * @todo: Add doxygen documentation
 */
class cumlHandle_impl {
public:
    cumlHandle_impl();
    ~cumlHandle_impl();
    int getDevice() const;
    void setStream( cudaStream_t stream );
    cudaStream_t getStream() const;
    void setDeviceAllocator( std::shared_ptr<deviceAllocator> allocator );
    std::shared_ptr<deviceAllocator> getDeviceAllocator() const;
    void setHostAllocator( std::shared_ptr<hostAllocator> allocator );
    std::shared_ptr<hostAllocator> getHostAllocator() const;

    cublasHandle_t getCublasHandle() const;
    cusolverDnHandle_t getcusolverDnHandle() const;
    cusparseHandle_t getcusparseHandle() const;

    cudaStream_t getInternalStream( int sid ) const;
    int getNumInternalStreams() const;

    void waitOnUserStream() const;
    void waitOnInternalStreams() const;

private:
    //TODO: What is the right number?
    static constexpr int                _num_streams = 3;
    const int                           _dev_id;
    std::vector<cudaStream_t>           _streams;
    cublasHandle_t                      _cublas_handle;
    cusolverDnHandle_t                  _cusolverDn_handle;
    cusparseHandle_t                    _cusparse_handle;
    std::shared_ptr<deviceAllocator>    _deviceAllocator;
    std::shared_ptr<hostAllocator>      _hostAllocator;
    cudaStream_t                        _userStream;
    cudaEvent_t                         _event;

    void createResources();
    void destroyResources();
};

namespace detail {

/**
 * @todo: Add doxygen documentation
 */
class streamSyncer {
public:
    streamSyncer( const cumlHandle_impl& handle )
        : _handle( handle )
    {
        _handle.waitOnUserStream();
    }
    ~streamSyncer()
    {
        _handle.waitOnInternalStreams();
    }

    streamSyncer(const streamSyncer& other) = delete;
    streamSyncer& operator=(const streamSyncer& other) = delete;
private:
    const cumlHandle_impl& _handle;
};

} // end namespace detail

} // end namespace ML
