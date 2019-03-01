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

class cumlHandle_impl {
public:
    cumlHandle_impl();
    ~cumlHandle_impl();
    void setStream( cudaStream_t stream );
    cudaStream_t getStream() const;
    void setDeviceAllocator( std::shared_ptr<deviceAllocator> allocator );
    std::shared_ptr<deviceAllocator> getDeviceAllocator() const;
    void setHostAllocator( std::shared_ptr<hostAllocator> allocator );
    std::shared_ptr<hostAllocator> getHostAllocator() const;

    cublasHandle_t getCublasHandle( int dev_idx = 0 ) const;
    cusolverDnHandle_t getcusolverDnHandle( int dev_idx = 0 ) const;
    cusparseHandle_t getcusparseHandle( int dev_idx ) const;

    cudaStream_t getInternalStream( int sid ) const;
    int getNumInternalStreams() const;

    void waitOnUserStream() const;
    void waitOnInternalStreams() const;

    void setDevice( int dev_id );
    void setDevices( const std::vector<int>& dev_ids );
    int  getDevice( int dev_idx = 0 ) const;
    int  getNumDevices() const;

    cudaStream_t getDeviceStream( int dev_idx ) const;
private:
    //TODO: What is the right number?
    static constexpr int                _num_streams = 3;
    std::vector<int>                    _dev_ids;
    std::vector<cudaStream_t>           _streams;
    std::vector<cublasHandle_t>         _cublas_handles;
    std::vector<cusolverDnHandle_t>     _cusolverDn_handles;
    std::vector<cusparseHandle_t>       _cusparse_handles;
    std::shared_ptr<deviceAllocator>    _deviceAllocator;
    std::shared_ptr<hostAllocator>      _hostAllocator;
    cudaStream_t                        _userStream;
    cudaEvent_t                         _event;
    
    void createResources();
    void destroyResources();
};

namespace detail {

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
