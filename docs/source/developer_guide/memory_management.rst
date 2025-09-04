=========================================
Memory Management Best Practices in cuML
=========================================

Introduction
------------
cuML is a GPU-accelerated machine learning library built on top of CUDA and RAPIDS.
To achieve high performance and reliability, cuML relies on **RAPIDS Memory Manager (RMM)**
for all device memory allocations.

Historically, some parts of cuML used **raw memory allocations** (e.g., calling
``mr->allocate()`` and ``mr->deallocate()`` directly). While this approach works,
it introduces significant risks:

- **Memory leaks**: forgetting to free memory after use.
- **Dangling pointers**: using freed memory by mistake.
- **Incompatibility**: does not integrate well with=========================================
Memory Management Best Practices in cuML
=========================================

Introduction
------------
cuML is a GPU-accelerated machine learning library built on top of CUDA and RAPIDS.
To achieve high performance and reliability, cuML relies on **RAPIDS Memory Manager (RMM)**
for all device memory allocations.

Historically, some parts of cuML used **raw memory allocations** (e.g., calling
``mr->allocate()`` and ``mr->deallocate()`` directly). While this approach works,
it introduces significant risks:

- **Memory leaks**: forgetting to free memory after use.
- **Dangling pointers**: using freed memory by mistake.
- **Incompatibility**: does not integrate well with modern CUDA allocators
  like ``device_async_resource_ref``.
- **Complexity**: manual bookkeeping across streams and scopes.
- **Inconsistency**: different RAPIDS libraries using different approaches.

To improve maintainability, all new cuML code should use **RMM containers**
such as ``rmm::device_buffer``, ``rmm::device_uvector``, and ``rmm::device_scalar``.

GPU Memory Management Challenges
--------------------------------
Managing GPU memory differs from CPU memory management because:

- GPU allocations are more expensive than CPU allocations.
- Memory must be tied to specific CUDA streams for correct synchronization.
- Allocations can fail if memory is fragmented, even when total free memory is sufficient.
- Debugging device memory issues is significantly harder than debugging CPU memory leaks.

Using RMM helps abstract away these complexities.

Overview of RMM
---------------
RAPIDS Memory Manager (RMM) provides:

- A **memory resource interface** (``rmm::mr::device_memory_resource``) that can
  be backed by different allocators (pool allocator, arena allocator, etc.).
- A set of **RAII containers** that automatically manage device memory and integrate
  with CUDA streams.

RMM ensures consistent memory handling across RAPIDS libraries such as cuDF, cuGraph, and cuML.

Raw Allocation vs. RMM
----------------------
**❌ Raw Allocation Example:**
.. code-block:: cpp

    // Allocate manually
    DataT* beta = static_cast<DataT*>(mr->allocate(sizeof(DataT) * n, stream));
    // ... use beta ...
    mr->deallocate(beta, sizeof(DataT) * n, stream);

- Must remember to deallocate explicitly.
- Error-prone and verbose.
- No integration with RAII (Resource Acquisition Is Initialization).

**✅ RMM uvector Example:**
.. code-block:: cpp

    rmm::device_uvector<DataT> beta(n, stream, mr);
    // Automatically freed when going out of scope.

- No manual deallocation required.
- Cleaner, safer code.
- Works across all RAPIDS libraries.

Recommended RMM Containers 
--------------------------
**1. rmm::device_buffer**
   - General-purpose, untyped device memory buffer.
   - Use when you need raw bytes of storage without type information.

   .. code-block:: cpp

       rmm::device_buffer buf(1024, stream, mr);

**2. rmm::device_uvector<T>**
   - Typed, resizable device array.
   - Ideal for storing vectors of data on the GPU.

   .. code-block:: cpp

       rmm::device_uvector<float> arr(n, stream, mr);

**3. rmm::device_scalar<T>**
   - Stores a single value in device memory.
   - Useful for reductions or scalars needed on the GPU.

   .. code-block:: cpp

       rmm::device_scalar<int> counter(0, stream, mr);

Migration Strategy
------------------
When contributing to cuML:

1. Search for raw pointer allocations in the codebase:
   ``DataT* ptr = static_cast<DataT*>(mr->allocate(...))``.
2. Replace them with RMM containers:
   - ``device_uvector`` for arrays.
   - ``device_buffer`` for untyped storage.
   - ``device_scalar`` for scalars.
3. Remove explicit ``mr->deallocate()`` calls.
4. Ensure that containers are tied to the correct CUDA stream.

Example Migration
-----------------
**Before:**
.. code-block:: cpp

    // Raw allocation
    double* tmp = static_cast<double*>(mr->allocate(n * sizeof(double), stream));
    my_kernel<<<grid, block, 0, stream>>>(tmp);
    mr->deallocate(tmp, n * sizeof(double), stream);

**After:**
.. code-block:: cpp

    // RMM-managed memory
    rmm::device_uvector<double> tmp(n, stream, mr);
    my_kernel<<<grid, block, 0, stream>>>(tmp.data());

Best Practices
--------------
- Always prefer RMM containers over raw pointers.
- Pass ``stream`` and ``mr`` explicitly when constructing RMM objects.
- Avoid implicit default streams to ensure reproducibility.
- Use ``device_scalar`` instead of allocating arrays of size 1.
- Free host-pinned memory explicitly if used, since RMM focuses on device memory.

Common Pitfalls
---------------
- **Mixing raw pointers with RMM**: can lead to double-free or leaks.
- **Forgetting to pass a stream**: can cause race conditions.
- **Using device_buffer when type information is needed**: prefer ``device_uvector<T>``.

References
----------
- RAPIDS Memory Manager (RMM): https://github.com/rapidsai/rmm
- cuML GitHub repository: https://github.com/rapidsai/cuml
- RAPIDS Developer Guide: https://docs.rapids.ai