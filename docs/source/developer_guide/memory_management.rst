Memory management in cuML
=========================

cuML uses the RAPIDS Memory Manager (RMM) to manage GPU memory.
Developers contributing to cuML should **always use RMM utilities**
instead of raw CUDA allocation (``cudaMalloc`` / ``cudaFree``).

Why RMM?
--------

- Avoids memory leaks and dangling pointers.
- Provides RAII-based containers (``device_uvector``, ``device_buffer``).
- Works consistently across all RAPIDS libraries.

Recommended containers
----------------------

- **rmm::device_uvector** – Typed, resizable GPU vector with RAII semantics.
- **rmm::device_buffer** – Untyped GPU buffer for raw storage.

Example
-------

.. code-block:: cpp

    #include <rmm/device_uvector.hpp>

    void example(rmm::cuda_stream_view stream) {
        rmm::device_uvector<float> data(100, stream);
        // Memory automatically freed when leaving scope
    }

Migration notes
---------------

When contributing to cuML:

- Replace raw ``cudaMalloc/cudaFree`` with RMM containers.
- Prefer ``device_uvector`` for typed data.
- Refer to the official `RMM documentation <https://github.com/rapidsai/rmm>`_ for advanced features.