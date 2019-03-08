# cuML developer guide
This document summarizes rules and best practices for contributions to the cuML C++ component of rapidsai/cuml. This is a living document and contributions for clarifications or fixes and issue reports are highly welcome.

## General
Please start by reading [CONTRIBUTING.md](../CONTRIBUTING.md).

## Thread safety
cuML is thread safe so its functions can be called from multiple host threads if they use different handles.

The implementation of cuML is single threaded.

## Coding style

## Error handling
Call CUDA APIs via the provided helper macros `CUDA_CHECK`, `CUBLAS_CHECK` and `CUSOLVER_CHECK`. These macros take care of checking the return values of the used API calls and generate an exception when the command is not successful. If you need to avoid an exception, e.g. inside a destructor, use `CUDA_CHECK_NO_THROW`, `CUBLAS_CHECK_NO_THROW ` and `CUSOLVER_CHECK_NO_THROW ` (currently not available, see https://github.com/rapidsai/cuml/issues/229). These macros log the error but do not throw an exception.

## Logging
Add once https://github.com/rapidsai/cuml/issues/100 is addressed.

## Documentation
All external interfaces need to have a complete [doxygen](http://www.doxygen.nl) API documentation. This is also recommended for internal interfaces.

## Testing and Unit Testing
TODO: Add this

## Device and Host memory allocations
To enable `libcuml.so` users to control how memory for temporary data is allocated, allocate device memory using the allocator provided:
```cpp
template<typename T>
void foo(const ML::cumlHandle_impl& h, cudaStream_t stream, ... )
{
    T* temp_h = h.getDeviceAllocator()->allocate(n*sizeof(T), stream);
    ...
    h.getDeviceAllocator()->deallocate(temp_h, n*sizeof(T), stream);
}
```
The same rule applies to larger amounts of host heap memory:
```cpp
template<typename T>
void foo(const ML::cumlHandle_impl& h, cudaStream_t stream, ... )
{
    T* temp_h = h.getHostAllocator()->allocate(n*sizeof(T), stream);
    ...
    h.getHostAllocator()->deallocate(temp_h, n*sizeof(T), stream);
}
```
Small host memory heap allocations, e.g. as internally done by STL containers, are fine, e.g. an `std::vector` managing only a handful of integers.
Both the Host and the Device Allocators might allow asynchronous stream ordered allocation and deallocation. This can provide significant performance benefits so a stream always needs to be specified when allocating or deallocating (see [Asynchronous operations and stream ordering](# Asynchronous operations and stream ordering)). `ML::deviceAllocator` returns pinned device memory on the current device, while `ML::hostAllocator` returns host memory. A user of cuML can write customized allocators and pass them into cuML. If a cuML user does not provide custom allocators default allocators will be used. For `ML::deviceAllocator` the default is to use `cudaMalloc`/`cudaFree`. For `ML::hostAllocator` the default is to use `cudaMallocHost`/`cudaFreeHost`.
There are two simple container classes compatible with the allocator interface `MLCommon::device_buffer` available in `ml-prims/src/common/device_buffer.hpp` and `MLCommon::host_buffer` available in `ml-prims/src/common/host_buffer.hpp`. These allow to follow the [RAII idiom](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization) to avoid resources leaks and enable exception safe code. These containers also allow asynchronous allocation and deallocation using the `resize` and `release` member functions:
```cpp
template<typename T>
void foo(const ML::cumlHandle_impl& h, ..., cudaStream_t stream )
{
    ...
    MLCommon::device_buffer<T> temp( h.getDeviceAllocator(), stream, 0 )
    
    temp.resize(n, stream);
    kernelA<<<grid, block, 0, stream>>>(..., temp.data(), ...);
    kernelB<<<grid, block, 0, stream>>>(..., temp.data(), ...);
    temp.release(stream);
}
```
The motivation for `MLCommon::host_buffer` and `MLCommon::device_buffer` over using `std::vector` or `thrust::device_vector` (which would require thrust 1.9.4 or later) is to enable exception safe asynchronous allocation and deallocation following stream semantics with an explicit interface while avoiding the overhead of implicitly initializing the underlying allocation.
To use `ML::hostAllocator` with a STL container the header `src/common/allocatorAdapter.hpp` provides `ML::stdAllocatorAdapter`:
```cpp
template<typename T>
void foo(const ML::cumlHandle_impl& h, ..., cudaStream_t stream )
{
    ...
    std::vector<T,ML::stdAllocatorAdapter<T> > temp( n, val, ML::stdAllocatorAdapter<T>(h.getHostAllocator(), stream) )
    ...
}
```
If thrust 1.9.4 or later is avaiable for use in cuML a similar allocator can be provided for `thrust::device_vector`.
### Using Thrust [AllocationsThrust]
To ensure that thrust algorithms allocate temporary memory via the provided device memory allocator, use the `ML::thrustAllocatorAdapter` available in `src/common/allocatorAdapter.hpp` with the `thrust::cuda::par` execution policy:
```cpp
void foo(const ML::cumlHandle_impl& h, ..., cudaStream_t stream )
{
    ML::thrustAllocatorAdapter alloc( h.getDeviceAllocator(), stream );
    auto execution_policy = thrust::cuda::par(alloc).on(stream);
    thrust::for_each(execution_policy, ... );
}
```
The header `src/common/allocatorAdapter.hpp` also provides a helper function to create an execution policy:
```cpp
void foo(const ML::cumlHandle_impl& h, ... , cudaStream_t stream )
{
    auto execution_policy = ML::thrust_exec_policy(h.getDeviceAllocator(),stream);
    thrust::for_each(execution_policy->on(stream), ... );
}
```

## Asynchronous operations and stream ordering
All ML algorithms should be as asynchronous as possible avoiding the use of the default stream (aka as NULL or `0` stream). Implementations that require only one CUDA Stream should use the stream from `ML::cumlHandle_impl`:
```cpp
void foo(const ML::cumlHandle_impl& h, ...)
{
    cudaStream_t stream = h.getStream();
}
```
When multiple streams are needed, e.g. to manage a pipeline, use the internal streams available in `ML::cumlHandle_impl` (see [CUDA Resources](# CUDA Resources)). If multiple streams are used all operations still must be ordered according to `ML::cumlHandle::getStream()`. Before any operation in any of the internal CUDA streams is started, all previous work in `ML::cumlHandle::getStream()` must have completed. Any work enqueued in `ML::cumlHandle::getStream()` after a cuML function returns should not start before all work enqueued in the internal streams has completed. E.g. if a cuML algorithm is called like this: 
```cpp
void foo(const double* const srcdata, double* const result)
{
    ML::cumlHandle cumlHandle;

    cudaStream_t stream;
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );
    cumlHandle.setStream( stream );

    ...

    CUDA_CHECK( cudaMemcpyAsync( srcdata, h_srcdata.data(), n*sizeof(double), cudaMemcpyHostToDevice, stream ) );

    ML::algo(cumlHandle, dopredict, srcdata, result, ... );

    CUDA_CHECK( cudaMemcpyAsync( h_result.data(), result, m*sizeof(int), cudaMemcpyDeviceToHost, stream ) );

    ...
}
```
No work in any stream should start in `ML::algo` before the `cudaMemcpyAsync` in `stream` launched before the call to `ML::algo` is done. And all work in all streams used in `ML::algo` should be done before the `cudaMemcpyAsync` in `stream` launched after the call to `ML::algo` starts.

This can be ensured by introducing interstream dependencies with CUDA events and `cudaStreamWaitEvent`. For convenience, the header `cumlHandle.hpp` provides the class `ML::detail::streamSyncer` which lets all `ML::cumlHandle_impl` internal CUDA streams wait on `ML::cumlHandle::getStream()` in its constructor and in its destructor and lets `ML::cumlHandle::getStream()` wait on all work enqueued in the `ML::cumlHandle_impl` internal CUDA streams. Here is an example:
```cpp
void cumlAlgo(const ML::cumlHandle_impl& h, ...)
{
    ML::detail::streamSyncer _(handle);
}
```
This ensures the stream ordering behavior described above.

### Using Thrust
To ensure that thrust algorithms are executed in the intended stream the `thrust::cuda::par` execution policy should be used (see [Using Thrust](# AllocationsThrust) in [Device and Host memory allocations](# Device and Host memory allocations)).

## CUDA Resources
Do not create reusable CUDA resources directly in Implementations of ML algorithms. Instead, use the existing resources in `ML::cumlHandle_impl ` to avoid constant creation and deletion of reusable resources such as CUDA streams, CUDA events or library handles. Please file a feature request if a resource handle is missing in `ML::cumlHandle_impl `.
The resources can be obtained like this
```cpp
void foo(const ML::cumlHandle_impl& h, ...)
{
    cublasHandle_t cublasHandle = h.getCublasHandle();
    const int num_streams       = h.getNumInternalStreams();
    const int stream_idx        = ...
    cudaStream_t stream         = h.getInternalStream(stream_idx);
    ...
}
```

## Multi GPU
 
The multi GPU paradigm of cuML is **O**ne **P**rocess per **G**PU (OPG). Each algorithm should be implemented in a way that it can run with a single GPU without any dependencies to any communication library. A multi GPU implementation can assume the following:
* The user of cuML has initialized MPI and created a communicator that can be used by the ML algorithm.
* All processes in the MPI communicator call into the ML algorithm cooperatively.
* The used MPI is CUDA-aware, i.e. it is possible to directly pass device pointers to MPI

## C APIs

ML algorithms implemented in cuML should have C++ APIs that are easy to wrap in C. Use only C compatible types or objects that can be passed as opaque handles (like `cumlHandle_t`). Using templates is fine if those can be instantiated from a specialized C++ function with `extern "C"` linkage.
