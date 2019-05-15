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

#include "cumlHandle.hpp"

#include "../../src_prims/utils.h"

//TODO: Delete CUBLAS_CHECK and CUSOLVER_CHECK once
//      https://github.com/rapidsai/cuml/issues/239 is addressed
#define CUBLAS_CHECK(call)                                                     \
  {                                                                            \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                             \
      fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,         \
              __LINE__);                                                       \
      switch (err) {                                                           \
        case CUBLAS_STATUS_NOT_INITIALIZED:                                    \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_NOT_INITIALIZED");            \
          exit(1);                                                             \
        case CUBLAS_STATUS_ALLOC_FAILED:                                       \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_ALLOC_FAILED");               \
          exit(1);                                                             \
        case CUBLAS_STATUS_INVALID_VALUE:                                      \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_INVALID_VALUE");              \
          exit(1);                                                             \
        case CUBLAS_STATUS_ARCH_MISMATCH:                                      \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_ARCH_MISMATCH");              \
          exit(1);                                                             \
        case CUBLAS_STATUS_MAPPING_ERROR:                                      \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_MAPPING_ERROR");              \
          exit(1);                                                             \
        case CUBLAS_STATUS_EXECUTION_FAILED:                                   \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_EXECUTION_FAILED");           \
          exit(1);                                                             \
        case CUBLAS_STATUS_INTERNAL_ERROR:                                     \
          fprintf(stderr, "%s\n", "CUBLAS_STATUS_INTERNAL_ERROR");             \
      }                                                                        \
      exit(1);                                                                 \
      exit(1);                                                                 \
    }                                                                          \
  }
#define CUSOLVER_CHECK(call)                                                   \
  {                                                                            \
    cusolverStatus_t err;                                                      \
    if ((err = (call)) != CUSOLVER_STATUS_SUCCESS) {                           \
      fprintf(stderr, "Got CUSOLVER error %d at %s:%d\n", err, __FILE__,       \
              __LINE__);                                                       \
      switch (err) {                                                           \
        case CUSOLVER_STATUS_NOT_INITIALIZED:                                  \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_NOT_INITIALIZED");          \
          exit(1);                                                             \
        case CUSOLVER_STATUS_ALLOC_FAILED:                                     \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_ALLOC_FAILED");             \
          exit(1);                                                             \
        case CUSOLVER_STATUS_INVALID_VALUE:                                    \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_INVALID_VALUE");            \
          exit(1);                                                             \
        case CUSOLVER_STATUS_ARCH_MISMATCH:                                    \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_ARCH_MISMATCH");            \
          exit(1);                                                             \
        case CUSOLVER_STATUS_MAPPING_ERROR:                                    \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_MAPPING_ERROR");            \
          exit(1);                                                             \
        case CUSOLVER_STATUS_EXECUTION_FAILED:                                 \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_EXECUTION_FAILED");         \
          exit(1);                                                             \
        case CUSOLVER_STATUS_INTERNAL_ERROR:                                   \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_INTERNAL_ERROR");           \
          exit(1);                                                             \
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                        \
          fprintf(stderr, "%s\n",                                              \
                  "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED");                \
          exit(1);                                                             \
        case CUSOLVER_STATUS_NOT_SUPPORTED:                                    \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_NOT_SUPPORTED");            \
          exit(1);                                                             \
        case CUSOLVER_STATUS_ZERO_PIVOT:                                       \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_ZERO_PIVOT");               \
          exit(1);                                                             \
        case CUSOLVER_STATUS_INVALID_LICENSE:                                  \
          fprintf(stderr, "%s\n", "CUSOLVER_STATUS_INVALID_LICENSE");          \
          exit(1);                                                             \
      }                                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }
#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      fprintf(stderr, "Got CUSPARSE error %d at %s:%d\n", err, __FILE__,       \
              __LINE__);                                                       \
      switch (err) {                                                           \
        case CUSPARSE_STATUS_NOT_INITIALIZED:                                  \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_NOT_INITIALIZED");          \
          exit(1);                                                             \
        case CUSPARSE_STATUS_ALLOC_FAILED:                                     \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_ALLOC_FAILED");             \
          exit(1);                                                             \
        case CUSPARSE_STATUS_INVALID_VALUE:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_INVALID_VALUE");            \
          exit(1);                                                             \
        case CUSPARSE_STATUS_ARCH_MISMATCH:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_ARCH_MISMATCH");            \
          exit(1);                                                             \
        case CUSPARSE_STATUS_MAPPING_ERROR:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_MAPPING_ERROR");            \
          exit(1);                                                             \
        case CUSPARSE_STATUS_EXECUTION_FAILED:                                 \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_EXECUTION_FAILED");         \
          exit(1);                                                             \
        case CUSPARSE_STATUS_INTERNAL_ERROR:                                   \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_INTERNAL_ERROR");           \
          exit(1);                                                             \
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                        \
          fprintf(stderr, "%s\n",                                              \
                  "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");                \
          exit(1);                                                             \
      }                                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace ML {

cumlHandle::cumlHandle() : _impl( new cumlHandle_impl() ) {}
cumlHandle::~cumlHandle(){}

void cumlHandle::setStream( cudaStream_t stream )
{
    _impl->setStream( stream );
}

cudaStream_t cumlHandle::getStream() const
{
    return _impl->getStream();
}

void cumlHandle::setDeviceAllocator( std::shared_ptr<deviceAllocator> allocator )
{
    _impl->setDeviceAllocator( allocator );
}

std::shared_ptr<deviceAllocator> cumlHandle::getDeviceAllocator() const
{
    return _impl->getDeviceAllocator();
}

void cumlHandle::setHostAllocator( std::shared_ptr<hostAllocator> allocator )
{
    _impl->setHostAllocator( allocator );
}

std::shared_ptr<hostAllocator> cumlHandle::getHostAllocator() const
{
    return _impl->getHostAllocator();
}

const cumlHandle_impl& cumlHandle::getImpl() const
{
    return *_impl.get();
}

using MLCommon::defaultDeviceAllocator;
using MLCommon::defaultHostAllocator;

cumlHandle_impl::cumlHandle_impl()
    : _dev_id( []() -> int { int cur_dev = -1; CUDA_CHECK( cudaGetDevice ( &cur_dev ) ); return cur_dev; }() ),
      _deviceAllocator( std::make_shared<defaultDeviceAllocator>() ), _hostAllocator( std::make_shared<defaultHostAllocator>() ),
      _userStream(NULL)
{
    createResources();
}

cumlHandle_impl::~cumlHandle_impl()
{
    destroyResources();
}

int cumlHandle_impl::getDevice() const
{
    return _dev_id;
}

void cumlHandle_impl::setStream( cudaStream_t stream )
{
    _userStream = stream;
}

cudaStream_t cumlHandle_impl::getStream() const
{
    return _userStream;
}

void cumlHandle_impl::setDeviceAllocator( std::shared_ptr<deviceAllocator> allocator )
{
    _deviceAllocator = allocator;
}

std::shared_ptr<deviceAllocator> cumlHandle_impl::getDeviceAllocator() const
{
    return _deviceAllocator;
}

void cumlHandle_impl::setHostAllocator( std::shared_ptr<hostAllocator> allocator )
{
    _hostAllocator = allocator;
}

std::shared_ptr<hostAllocator> cumlHandle_impl::getHostAllocator() const
{
    return _hostAllocator;
}

cublasHandle_t cumlHandle_impl::getCublasHandle() const
{
    return _cublas_handle;
}

cusolverDnHandle_t cumlHandle_impl::getcusolverDnHandle() const
{
    return _cusolverDn_handle;
}

cusparseHandle_t cumlHandle_impl::getcusparseHandle() const
{
    return _cusparse_handle;
}

cudaStream_t cumlHandle_impl::getInternalStream( int sid ) const
{
    return _streams[sid];
}

int cumlHandle_impl::getNumInternalStreams() const
{
    return _num_streams;
}

void cumlHandle_impl::waitOnUserStream() const
{
    CUDA_CHECK( cudaEventRecord( _event, _userStream ) );
    for (auto s : _streams)
    {
        CUDA_CHECK( cudaStreamWaitEvent( s, _event, 0 ) );
    }
}

void cumlHandle_impl::waitOnInternalStreams() const
{
    for (auto s : _streams)
    {
        CUDA_CHECK( cudaEventRecord( _event, s ) );
        CUDA_CHECK( cudaStreamWaitEvent( _userStream, _event, 0 ) );
    }
}

void cumlHandle_impl::createResources()
{
    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate(&stream) );

    CUBLAS_CHECK( cublasCreate(&_cublas_handle) );

    CUSOLVER_CHECK( cusolverDnCreate(&_cusolverDn_handle) );

    CUSPARSE_CHECK( cusparseCreate(&_cusparse_handle) );

    _streams.push_back(stream);
    for (int i = 1; i < _num_streams; ++i)
    {
        cudaStream_t stream;
        CUDA_CHECK( cudaStreamCreate(&stream) );
        _streams.push_back(stream);
    }
    CUDA_CHECK( cudaEventCreateWithFlags( &_event, cudaEventDisableTiming ) );
}

void cumlHandle_impl::destroyResources()
{
    {
        cusparseStatus_t status = cusparseDestroy( _cusparse_handle );
        if ( CUSPARSE_STATUS_SUCCESS != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUSPARSE_CHECK is not used.
        }
    }

    {
        cusolverStatus_t status = cusolverDnDestroy( _cusolverDn_handle );
        if ( CUSOLVER_STATUS_SUCCESS != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUSOLVER_CHECK is not used.
        }
    }

    {
        cublasStatus_t status = cublasDestroy( _cublas_handle );
        if ( CUBLAS_STATUS_SUCCESS != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUBLAS_CHECK is not used.
        }
    }

    while ( !_streams.empty() )
    {
        cudaError_t status = cudaStreamDestroy( _streams.back() );
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
        _streams.pop_back();
    }
    cudaError_t status = cudaEventDestroy( _event );
    if ( cudaSuccess != status )
    {
        //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
        // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
    }
}

HandleMap handleMap;

std::pair<cumlHandle_t, cumlError_t> HandleMap::createAndInsertHandle()
{
    cumlError_t status = CUML_SUCCESS;
	  cumlHandle_t chosen_handle;
    try
    {
        auto handle_ptr = new ML::cumlHandle();
        bool inserted;
        {
            std::lock_guard<std::mutex> guard(_mapMutex);
            cumlHandle_t initial_next = _nextHandle;
            do {
                // try to insert using next free handle identifier
                chosen_handle = _nextHandle;
                inserted = _handleMap.insert({ chosen_handle, handle_ptr}).second;
                _nextHandle += 1;
            } while(!inserted && _nextHandle != initial_next);
        }
        if (!inserted) {
            // no free handle identifier available
            chosen_handle = INVALID_HANDLE;
            status = CUML_ERROR_UNKNOWN;
        }
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKNOWN;
        chosen_handle = CUML_ERROR_UNKNOWN;
    }
    return std::pair<cumlHandle_t, cumlError_t>(chosen_handle, status);
}

std::pair<cumlHandle*, cumlError_t> HandleMap::lookupHandlePointer(cumlHandle_t handle) const
{
    std::lock_guard<std::mutex> guard(_mapMutex);
    auto it = _handleMap.find(handle);
    if (it == _handleMap.end()) {
        return std::pair<cumlHandle*, cumlError_t>(nullptr, CUML_INVALID_HANDLE);
    } else {
        return std::pair<cumlHandle*, cumlError_t>(it->second, CUML_SUCCESS);
    }
}

cumlError_t HandleMap::removeAndDestroyHandle(cumlHandle_t handle)
{
    ML::cumlHandle *handle_ptr;
    {
        std::lock_guard<std::mutex> guard(_mapMutex);
        auto it = _handleMap.find(handle);
        if (it == _handleMap.end()) {
            return CUML_INVALID_HANDLE;
        }
        handle_ptr = it->second;
        _handleMap.erase(it);
    }
    cumlError_t status = CUML_SUCCESS;
    try
    {
        delete handle_ptr;
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKNOWN;
    }
    return status;
}

} // end namespace ML
