# cuML developer guide
This document summarizes rules and best practices for contributions to the cuML C++ component of rapidsai/cuml. This is a living document and contributions for clarifications or fixes and issue reports are highly welcome.

## General
Please start by reading [CONTRIBUTING.md](../CONTRIBUTING.md).

## Thread safety
cuML is thread safe so its functions can be called from multiple host threads if they use different handles.

The implementation of cuML is single threaded.

## Managing state
### Introduction
Every ML algo needs to store some state, eg: model and its related hyper-parameters. Thus, this section lays out guidelines for managing state along the API of cuML.

### Inside `libcuml++.so` aka our C++ interface
Functions exposed via the cuML-C++ layer must be stateless. Meaning, they must accept all the required inputs, parameters and outputs in their argument list only. Things which are OK to be exposed on the interface:
1. Any [POD](https://en.wikipedia.org/wiki/Passive_data_structure).
2. `cumlHandle` - since it stores GPU-related state which has nothing to do with the model/algo state.
3. Pointers (explicitly putting it out, even though can be considered as a POD).

Taking dbscan algo as an example, the right way to expose the interface from `libcuml++.so` is:
```cpp
void dbscanFit(const cumlHandle &handle, const float *input, int n_rows, int n_cols,
               float eps, int min_pts, int *labels, size_t max_bytes_per_batch=0,
               bool verbose=false);
```

However, the following way of exposing dbscan API would be wrong. Because, the developer has exposed a C++ class with state along the interface of `libcuml++.so`.
```cpp
class DbscanClass {
public:
  DbscanClass(const cumlHandle& handle, float eps, int min_pts,
              size_t max_bytes_per_batch=0, bool verbose=false);
  void fit(const float *input, int n_rows, int n_cols, int *labels);
};
void dbscanFit(DbscanClass& dbscan, const float *input, int n_rows, int n_cols,
               int *labels);
```
In other words, the stateful API should always be a wrapper around the stateless methods, NEVER the other way around.

That said, internally, these stateless functions are free to create their own temporary classes, as long as they are not exposed on the interface of `libcuml++.so`. For example, one way to fix the above state-based interface could be to remove the `DbscanClass` declaration from the interface header and move it inside the internals of cuML. Then, create its object only inside the `dbscanFit` method.
```cpp
void dbscanFit(const cumlHandle &handle, const float *input, int n_rows, int n_cols,
               float eps, int min_pts, int *labels, size_t max_bytes_per_batch=0,
               bool verbose=false) {
  DbscanClass d(eps, min_pts, max_bytes_per_batch, verbose);
  d.fit(input, n_rows, n_cols, labels);
}
```

### scikit-learn-esq stateful API in C++
We are [still discussing](https://github.com/rapidsai/cuml/issues/456) about the right way to expose such a wrapper API around `libcuml++.so`. Stay tuned for more details.

## Coding style

## Code format
### Introduction
cuML relies on `clang-format` to enforce code style across all C++ and CUDA source code. The coding style is based on the [Google style guide](https://google.github.io/styleguide/cppguide.html#Formatting). The only digressions from this style are the following.
1. Do not split empty functions/records/namespaces.
2. Two-space indentation everywhere, including the line continuations.
3. Disable reflowing of comments.
The reasons behind these deviations from the Google style guide are given in comments [here](./.clang-format).

### How is the check done?
[run-clang-format.py](scripts/run-clang-format.py) is run first by `make`. This script runs clang-format only on modified files. An error is raised if the code diverges from the format suggested by clang-format, and the build fails.

### How to know the formatting violations?
When there are formatting errors, `run-clang-format.py` prints a `diff` command, showing where there are formatting differences. Unfortunately, unlike `flake8`, `clang-format` does NOT print descriptions of the violations, but instead directly formats the code. So, the only way currently to know why there are formatting differences is to run the diff command as suggested by this script against each violating source file.

### How to fix the formatting violations?
When there are formatting violations, `run-clang-format.py` prints an `-inplace` command you can use to automatically fix formatting errors. This is the easiest way to fix formatting errors. [This screencast](https://asciinema.org/a/248215) shows a typical build-fix-build cycle during cuML development.

### clang-format version?
To avoid spurious code style violations we specify the exact clang-format version required, currently `8.0.0`. This is enforced by a CMake check for the required version. [See here for more details on the dependencies](./README.md#dependencies).

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
Both the Host and the Device Allocators might allow asynchronous stream ordered allocation and deallocation. This can provide significant performance benefits so a stream always needs to be specified when allocating or deallocating (see [Asynchronous operations and stream ordering](#asynchronous-operations-and-stream-ordering)). `ML::deviceAllocator` returns pinned device memory on the current device, while `ML::hostAllocator` returns host memory. A user of cuML can write customized allocators and pass them into cuML. If a cuML user does not provide custom allocators default allocators will be used. For `ML::deviceAllocator` the default is to use `cudaMalloc`/`cudaFree`. For `ML::hostAllocator` the default is to use `cudaMallocHost`/`cudaFreeHost`.
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

### <a name="allocationsthrust"></a>Using Thrust
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
When multiple streams are needed, e.g. to manage a pipeline, use the internal streams available in `ML::cumlHandle_impl` (see [CUDA Resources](#cuda-resources)). If multiple streams are used all operations still must be ordered according to `ML::cumlHandle::getStream()`. Before any operation in any of the internal CUDA streams is started, all previous work in `ML::cumlHandle::getStream()` must have completed. Any work enqueued in `ML::cumlHandle::getStream()` after a cuML function returns should not start before all work enqueued in the internal streams has completed. E.g. if a cuML algorithm is called like this: 
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

This can be ensured by introducing interstream dependencies with CUDA events and `cudaStreamWaitEvent`. For convenience, the header `cumlHandle.hpp` provides the class `ML::detail::streamSyncer` which lets all `ML::cumlHandle_impl` internal CUDA streams wait on `ML::cumlHandle::getStream()` in its constructor and in its destructor and lets `ML::cumlHandle::getStream()` wait on all work enqueued in the `ML::cumlHandle_impl` internal CUDA streams. The intended use would be to create a `ML::detail::streamSyncer` object as the first thing in a entry function of the public cuML API: 

```cpp
void cumlAlgo(const ML::cumlHandle& handle, ...)
{
    ML::detail::streamSyncer _(handle.getImpl());
}
```
This ensures the stream ordering behavior described above.

### Using Thrust
To ensure that thrust algorithms are executed in the intended stream the `thrust::cuda::par` execution policy should be used (see [Using Thrust](#allocationsthrust) in [Device and Host memory allocations](#device-and-host-memory-allocations)).

## CUDA Resources

Do not create reusable CUDA resources directly in implementations of ML algorithms. Instead, use the existing resources in `ML::cumlHandle_impl` to avoid constant creation and deletion of reusable resources such as CUDA streams, CUDA events or library handles. Please file a feature request if a resource handle is missing in `ML::cumlHandle_impl `.
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

### `ML::cumlHandle` and `ML::cumlHandle_impl`

The purpose of `ML::cumlHandle` is to be the public interface of cuML, i.e. it is meant to be used by developers using cuML in their application. This is differentiated from `ML::cumlHandle_impl` to avoid that the public interface of cuML depends on cuML internals, such as CUDA library handles, e.g. for cuBLAS. This is implemented via the "Pointer to implementation" or [pImpl](https://en.cppreference.com/w/cpp/language/pimpl) idiom. From a `ML::cumlHandle` the implementation `ML::cumlHandle_impl` can be obtained by calling `ML::cumlHandle::getImpl()`. The implementation of cuML should use `ML::cumlHandle_impl` and not `ML::cumlHandle`. E.g. for the function `ml_algo` from the public cuML interface an implementation calling the internal functions `foo` and `bar` could look like this:

```cpp
void ml_algo(const ML::cumlHandle& handle, ...)
{
    const ML::cumlHandle_impl& h = handle.getImpl();
    ML::detail::streamSyncer _(h);
    ...
    foo(h, ...);
    ...
    bar(h, ...);
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
