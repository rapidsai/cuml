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

#include "../../../ml-prims/src/utils.h"

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

const cumlHandle_impl* cumlHandle::getImpl() const
{
    return _impl.get();
}

class cudaDeviceScope {
public:
    cudaDeviceScope( const int dev_id )
        : _dev_id( dev_id )
    {
        CUDA_CHECK( cudaGetDevice(&_old_dev_id) );
        if ( _dev_id != _old_dev_id ) {
            CUDA_CHECK( cudaSetDevice(_dev_id) );
        }
    }
    ~cudaDeviceScope()
    {
        if ( _dev_id != _old_dev_id ) {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            cudaSetDevice(_old_dev_id);
        }
    }
private:
    const int _dev_id;
    int _old_dev_id;
};

class defaultDeviceAllocator : public deviceAllocator {
public:
    virtual void* allocate( std::size_t n, cudaStream_t ) {
        void* ptr = 0;
        CUDA_CHECK( cudaMalloc( &ptr, n ) );
        return ptr;
    }
    virtual void deallocate( void* p, std::size_t, cudaStream_t ) {
        cudaError_t status = cudaFree( p);
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
    }
};

class defaultHostAllocator : public hostAllocator {
public:
    virtual void* allocate( std::size_t n, cudaStream_t ) {
        void* ptr = 0;
        CUDA_CHECK( cudaMallocHost( &ptr, n ) );
        return ptr;
    }
    virtual void deallocate( void* p, std::size_t, cudaStream_t ) {
        cudaError_t status = cudaFreeHost( p);
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
    }
};

//TODO provide deviceAllocator adaptor for a pair of function pointers (needed for C interface)

cumlHandle_impl::cumlHandle_impl()
    : _dev_ids(1,0), _deviceAllocator( std::make_shared<defaultDeviceAllocator>() ), _hostAllocator( std::make_shared<defaultHostAllocator>() )
{
    createResources();
}

cumlHandle_impl::~cumlHandle_impl()
{
    destroyResources();
}

void cumlHandle_impl::setDevice( int dev_id )
{
    destroyResources();
    _dev_ids[0] = dev_id;
    createResources();
}

void cumlHandle_impl::setDevices(const std::vector<int>& dev_ids )
{
    destroyResources();
    _dev_ids = dev_ids;
    createResources();
}

int cumlHandle_impl::getDevice( int dev_idx ) const
{
    return _dev_ids[dev_idx];
}

int cumlHandle_impl::getNumDevices() const
{
    return _dev_ids.size();
}

void cumlHandle_impl::setStream( cudaStream_t stream )
{
    _userStream = stream;
}

cudaStream_t cumlHandle_impl::getStream() const
{
    return _userStream;
}

cudaStream_t cumlHandle_impl::getDeviceStream( int dev_idx ) const
{
    return _streams[dev_idx*_num_streams + 0];
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

cublasHandle_t cumlHandle_impl::getCublasHandle( int dev_idx ) const
{
    return _cublas_handles[dev_idx];
}

cusolverDnHandle_t cumlHandle_impl::getcusolverDnHandle( int dev_idx ) const
{
    return _cusolverDn_handles[dev_idx];
}

cusparseHandle_t cumlHandle_impl::getcusparseHandle( int dev_idx ) const
{
    return _cusparse_handles[dev_idx];
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
    CUDA_CHECK( cudaEventCreateWithFlags( &_event, cudaEventDisableTiming ) );
    cudaDeviceScope _( 0 );
    for (auto dev_id: _dev_ids) 
    {
        CUDA_CHECK( cudaSetDevice(dev_id) );

        cudaStream_t stream;
        CUDA_CHECK( cudaStreamCreate(&stream) );

        cublasHandle_t cublas_handle;
        CUBLAS_CHECK( cublasCreate(&cublas_handle) );
        CUBLAS_CHECK( cublasSetStream(cublas_handle, stream) );

        cusolverDnHandle_t cusolverDn_handle;
        CUSOLVER_CHECK( cusolverDnCreate(&cusolverDn_handle) );
        CUSOLVER_CHECK( cusolverDnSetStream(cusolverDn_handle, stream) );

        cusparseHandle_t cusparse_handle;
        CUSPARSE_CHECK( cusparseCreate(&cusparse_handle) );
        CUSPARSE_CHECK( cusparseSetStream(cusparse_handle, stream) );

        _streams.push_back(stream);
        _cublas_handles.push_back( cublas_handle );
        _cusolverDn_handles.push_back( cusolverDn_handle );
        _cusparse_handles.push_back( cusparse_handle );
        
        for (int i = 1; i < _num_streams; ++i)
        {
            cudaStream_t stream;
            CUDA_CHECK( cudaStreamCreate(&stream) );
            _streams.push_back(stream);
        }
    }
}

void cumlHandle_impl::destroyResources()
{
    while ( !_cusparse_handles.empty() )
    {
        cusparseStatus_t status = cusparseDestroy( _cusparse_handles.back() );
        if ( CUSPARSE_STATUS_SUCCESS != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUSPARSE_CHECK is not used.
        }
        _cusparse_handles.pop_back();
    }
    while ( !_cusolverDn_handles.empty() )
    {
        cusolverStatus_t  status = cusolverDnDestroy( _cusolverDn_handles.back() );
        if ( CUSOLVER_STATUS_SUCCESS != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUSOLVER_CHECK is not used.
        }
        _cusolverDn_handles.pop_back();
    }
    while ( !_cublas_handles.empty() )
    {
        cublasStatus_t status = cublasDestroy( _cublas_handles.back() );
        if ( CUBLAS_STATUS_SUCCESS != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUBLAS_CHECK is not used.
        }
        _cublas_handles.pop_back();
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

} // end namespace ML
