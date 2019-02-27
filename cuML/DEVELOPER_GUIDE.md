# cuML developer guide
This document summarizes rules and best practices for contributions to C++ component cuML of RAPIDS/cuml.

# General

# Thread safety

# Coding style

# Error handling
All calls to CUDA APIs should be done via the provided helper macros `CUDA_CHECK`, `CUBLAS_CHECK` and `CUSOLVER_CHECK`. Those macros take care of checking the return values of the used API calls and generate an exception in case the command was no successful. In case an exception needs to be avoided, e.g. when implementing a destructor, `CUDA_CHECK_NO_THROW`, `CUBLAS_CHECK_NO_THROW ` and `CUSOLVER_CHECK_NO_THROW ` (currently not available, see https://github.com/rapidsai/cuml/issues/229) should be used. Those macros will only log the error but do not throw an exception.

# Logging
Add once https://github.com/rapidsai/cuml/issues/100 is addressed.

# Device and Host memory allocations
To enable `libcuml.so` users to control how memory for temporary data is allocated device memory should only be allocated via the allocator provided:
```cpp
template<typename T>
void foo(ML::cumlHandle* handle, cudaStream_t stream, ... )
{
    T* temp_h = handle->getDeviceAllocator()->allocate(n*sizeof(T), stream);
    ...
    handle->getDeviceAllocator()->deallocate(temp_h, n*sizeof(T), stream);
}
```
the same rule applies to larger amounts of host heap memory:
```cpp
template<typename T>
void foo(ML::cumlHandle* handle, cudaStream_t stream, ... )
{
    T* temp_h = handle->getHostAllocator()->allocate(n*sizeof(T), stream);
    ...
    handle->getHostAllocator()->deallocate(temp_h, n*sizeof(T), stream);
}
```
Small host memory heap allocations, e.g. as internally done by STL containers, are fine, e.g. an `std::vector` managing only a handful of integers.
Both the Host and the Device Allocators might allow asynchronous stream ordered allocation and deallocation. This can provide significant performance benefits so a stream always needs to be specified when allocating or deallocating (see [Asynchronous operations and stream ordering](# Asynchronous operations and stream ordering)).
There are two simple container classes compatible with the `ML::cumlHandle` allocator interface `ML::device_buffer` available in ` cuML/src/common/device_buffer.hpp` and `ML::host_buffer` available in ` cuML/src/common/host_buffer.hpp`. These allow to follow the [RAII idiom](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization) to avoid resources leaks and enable exception safe code. These containers also allow asynchronous allocation and deallocation using the `resize` and `release` member functions:
```cpp
template<typename T>
void foo( ML::cumlHandle* handle, .., cudaStream_t stream )
{
    ...
    ML::device_buffer<T> temp( handle->getDeviceAllocator(), 0 )
    
    temp.resize(n, stream);
    kernelA<<<grid, block, 0, stream>>>(..., temp.data(), ...);
    kernelB<<<grid, block, 0, stream>>>(..., temp.data(), ...);
    temp.release(stream);
}
```
## Using thrust
To ensure that thrust algorithms allocate temporary memory via the provided device memory allocator the `ML::thrustAllocatorAdapter` available in `allocatorAdapter.hpp` should be used with the `thrust::cuda::par` execution policy:
```cpp
void foo( ML::cumlHandle* handle, .. )
{
    ML::thrustAllocatorAdapter alloc( handle->getDeviceAllocator(), handle->getStream() );
    auto execution_policy = thrust::cuda::par(alloc).on(handle->getStream());
    thrust::for_each(execution_policy, ... );
}
```
# Asynchronous operations and stream ordering
All ML algorithms should be as asynchronous as possible avoiding the use of the default stream (aka as NULL or `0` stream). If an implementation only requires a single CUDA Stream the stream from `ML::cumlHandle` should be used:
```cpp
void foo( ML::cumlHandle* handle, ...)
{
    cudaStream_t stream = handle->getStream();
}
```
In case multiple streams are needed, e.g. to manage a pipeline, the internal streams available in `ML::cumlHandle_impl` should be used (see [CUDA Resources](# CUDA Resources)). If multiple streams are used all operations still need to be ordered according to `ML::cumlHandle::getStream()`, i.e. before any operation in any of the internal CUDA streams is started all previous work in `ML::cumlHandle: getStream()` needs to have completed and any work enqueued in `ML::cumlHandle::getStream()` after an cuML function returns should not start before all work enqueued in the internal stream has complete. This can be ensured by introducing inter stream dependencies with CUDA events and `cudaStreamWaitEvent`. For convenience  the header `cumlHandle.hpp` provide the class `ML::detail::streamSyncer` which lets all `ML::cumlHandle_impl` internal CUDA streams wait on `ML::cumlHandle::getStream()` in its constructor and in its destructor and lets `ML::cumlHandle::getStream()` wait on all work enqueued in the `ML::cumlHandle_impl` internal CUDA streams. E.g. a usage like this:
```cpp
void cumlAlgo( ML::cumlHandle* handle, ...)
{
    ML::detail::streamSyncer _(handle);
}
```
ensures the stream ordering behavior described above.

## Using thrust
To ensure that thrust algorithms are executed in the intended stream the `thrust::cuda::par` execution policy should be used (described in the section Device and Host memory allocations/Using thrust).

# CUDA Resources
Implementations of ML algorithms should not create reusable resources themselves. Instead they should use the existing one in `ML::cumlHandle_impl `. This allows to avoid constant creating and recreation of reusable resources such as CUDA streams, CUDA events or library handles. Please file a feature request in case a resource handle is missing in `ML::cumlHandle_impl `.
The resources can be obtained like this
```cpp
void foo( ML::cumlHandle* handle, ...)
{
    cublasHandle_t cublasHandle = handle->getImpl()->getCublasHandle();
    ...
}
```

