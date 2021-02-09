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

#include <cuml/cuml_api.h>

#include <raft/cudart_utils.h>
#include <cuml/common/utils.hpp>
#include <functional>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/host/allocator.hpp>
#include "cumlHandle.hpp"

namespace ML {
namespace detail {

class hostAllocatorFunctionWrapper : public raft::mr::host::allocator {
 public:
  hostAllocatorFunctionWrapper(cuml_allocate allocate_fn,
                               cuml_deallocate deallocate_fn)
    : _allocate_fn(allocate_fn), _deallocate_fn(deallocate_fn) {}

  virtual void* allocate(std::size_t n, cudaStream_t stream) {
    void* ptr = 0;
    CUDA_CHECK(_allocate_fn(&ptr, n, stream));
    return ptr;
  }

  virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) {
    CUDA_CHECK_NO_THROW(_deallocate_fn(p, n, stream));
  }

 private:
  const std::function<cudaError_t(void**, size_t, cudaStream_t)> _allocate_fn;
  const std::function<cudaError_t(void*, size_t, cudaStream_t)> _deallocate_fn;
};

class deviceAllocatorFunctionWrapper
  : public raft::mr::device::default_allocator {
 public:
  deviceAllocatorFunctionWrapper(cuml_allocate allocate_fn,
                                 cuml_deallocate deallocate_fn)
    : _allocate_fn(allocate_fn), _deallocate_fn(deallocate_fn) {}

  virtual void* allocate(std::size_t n, cudaStream_t stream) {
    void* ptr = 0;
    CUDA_CHECK(_allocate_fn(&ptr, n, stream));
    return ptr;
  }

  virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) {
    CUDA_CHECK_NO_THROW(_deallocate_fn(p, n, stream));
  }

 private:
  const std::function<cudaError_t(void**, size_t, cudaStream_t)> _allocate_fn;
  const std::function<cudaError_t(void*, size_t, cudaStream_t)> _deallocate_fn;
};

}  // end namespace detail
}  // end namespace ML

extern "C" const char* cumlGetErrorString(cumlError_t error) {
  switch (error) {
    case CUML_SUCCESS:
      return "success";
    case CUML_ERROR_UNKNOWN:
      //Intentional fall through
    default:
      return "unknown";
  }
}

extern "C" cumlError_t cumlCreate(cumlHandle_t* handle) {
  cumlError_t status;
  std::tie(*handle, status) = ML::handleMap.createAndInsertHandle();
  return status;
}

extern "C" cumlError_t cumlSetStream(cumlHandle_t handle, cudaStream_t stream) {
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      handle_ptr->set_stream(stream);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

extern "C" cumlError_t cumlGetStream(cumlHandle_t handle,
                                     cudaStream_t* stream) {
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      *stream = handle_ptr->get_stream();
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

extern "C" cumlError_t cumlSetDeviceAllocator(cumlHandle_t handle,
                                              cuml_allocate allocate_fn,
                                              cuml_deallocate deallocate_fn) {
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      std::shared_ptr<ML::detail::deviceAllocatorFunctionWrapper> allocator(
        new ML::detail::deviceAllocatorFunctionWrapper(allocate_fn,
                                                       deallocate_fn));
      handle_ptr->set_device_allocator(allocator);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

extern "C" cumlError_t cumlSetHostAllocator(cumlHandle_t handle,
                                            cuml_allocate allocate_fn,
                                            cuml_deallocate deallocate_fn) {
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      std::shared_ptr<ML::detail::hostAllocatorFunctionWrapper> allocator(
        new ML::detail::hostAllocatorFunctionWrapper(allocate_fn,
                                                     deallocate_fn));
      handle_ptr->set_host_allocator(allocator);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

extern "C" cumlError_t cumlDestroy(cumlHandle_t handle) {
  return ML::handleMap.removeAndDestroyHandle(handle);
}
