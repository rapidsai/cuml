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

#include "cumlHandle.hpp"

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
