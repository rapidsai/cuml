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
#include "cuML_api.h"

#include <functional>

#include "cumlHandle.hpp"

#include "../../../ml-prims/src/utils.h"

namespace ML {
namespace detail {

class hostAllocatorFunctionWrapper : public MLCommon::hostAllocator
{
public:
    hostAllocatorFunctionWrapper(cuml_allocate allocate_fn, cuml_deallocate deallocate_fn)
        : _allocate_fn(allocate_fn), _deallocate_fn(deallocate_fn)
    {}

    virtual void* allocate(std::size_t n, cudaStream_t stream)
    {
        void* ptr = 0;
        CUDA_CHECK( _allocate_fn( &ptr, n, stream ) );
        return ptr;
    }

    virtual void deallocate(void* p, std::size_t n, cudaStream_t stream)
    {
        cudaError_t status = _deallocate_fn(p,n,stream);
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
    }

private:
    const std::function<cudaError_t(void**, size_t, cudaStream_t)> _allocate_fn;
    const std::function<cudaError_t(void*, size_t, cudaStream_t)> _deallocate_fn;
};

class deviceAllocatorFunctionWrapper : public MLCommon::deviceAllocator
{
public:
    deviceAllocatorFunctionWrapper(cuml_allocate allocate_fn, cuml_deallocate deallocate_fn)
        : _allocate_fn(allocate_fn), _deallocate_fn(deallocate_fn)
    {}

    virtual void* allocate(std::size_t n, cudaStream_t stream)
    {
        void* ptr = 0;
        CUDA_CHECK( _allocate_fn( &ptr, n, stream ) );
        return ptr;
    }

    virtual void deallocate(void* p, std::size_t n, cudaStream_t stream)
    {
        cudaError_t status = _deallocate_fn(p,n,stream);
        if ( cudaSuccess != status )
        {
            //TODO: Add loging of this error. Needs: https://github.com/rapidsai/cuml/issues/100
            // deallocate should not throw execeptions which is why CUDA_CHECK is not used.
        }
    }

private:
    const std::function<cudaError_t(void**, size_t, cudaStream_t)> _allocate_fn;
    const std::function<cudaError_t(void*, size_t, cudaStream_t)> _deallocate_fn;
};

} // end namespace detail
} // end namespace ML

extern "C" const char* cumlGetErrorString ( cumlError_t error )
{
    switch( error )
    {
        case CUML_SUCCESS:
            return "success";
        case CUML_ERROR_UNKOWN:
            //Intentional fall through
        default:
            return "unknown";
    }
}

extern "C" cumlError_t cumlCreate( cumlHandle_t* handle )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        handle->ptr = new ML::cumlHandle();
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t cumlSetStream( cumlHandle_t handle, cudaStream_t stream )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        reinterpret_cast<ML::cumlHandle*>(handle.ptr)->setStream( stream );
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t cumlGetStream( cumlHandle_t handle, cudaStream_t* stream )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        *stream = reinterpret_cast<ML::cumlHandle*>(handle.ptr)->getStream();
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t cumlSetDeviceAllocator( cumlHandle_t handle, cuml_allocate allocate_fn, cuml_deallocate deallocate_fn )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        std::shared_ptr<ML::detail::deviceAllocatorFunctionWrapper> allocator( new ML::detail::deviceAllocatorFunctionWrapper(allocate_fn,deallocate_fn) );
        reinterpret_cast<ML::cumlHandle*>(handle.ptr)->setDeviceAllocator(allocator);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}
extern "C" cumlError_t cumlSetHostAllocator( cumlHandle_t handle, cuml_allocate allocate_fn, cuml_deallocate deallocate_fn )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        std::shared_ptr<ML::detail::hostAllocatorFunctionWrapper> allocator( new ML::detail::hostAllocatorFunctionWrapper(allocate_fn,deallocate_fn) );
        reinterpret_cast<ML::cumlHandle*>(handle.ptr)->setHostAllocator(allocator);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t cumlDestroy( cumlHandle_t handle )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        delete reinterpret_cast<ML::cumlHandle*>(handle.ptr);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}
