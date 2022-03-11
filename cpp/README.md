# cuML C++

This folder contains the C++ and CUDA code of the algorithms and ML primitives of cuML. The build system uses CMake for build configuration, and an out-of-source build is recommended.

## Source Code Folders

The source code of cuML is divided in three main directories: `include`, `src`, and `src_prims`.

- `src` contains the source code of the Machine Learning algorithms, and the main cuML C++ API. The main consumable is the shared library `libcuml++`, that can be used stand alone by C++ consumers or is consumed by our Python package `cuml` to provide a Python API.
- `src_prims` contains most of the common components and computational primitives that form part of the machine learning algorithms in cuML, and can be used individually as well in the form of a header only library.
- `comms` contains the source code of the communications implementations that enable multi-node multi-GPU algorithms. There are currently two communications implementations. The implementation in the `mpi` directory is for MPI environments. It can also be used for automated tested. The implementation in the `std` directory is required for running cuML in multi-node multi-GPU Dask environments.

The `test` directory has subdirectories that reflect this distinction between the `src` and `prims` components of cuML.

## Setup and Dependencies

To build the C++ artifacts, please refer to the [build documentation](../BUILD.md).

## Using cuML libraries

After building cuML, you can use its functionality in other C/C++ applications
by linking against the generated libraries. The following trivial example shows
how to make external use of cuML's logger:

```cpp
// main.cpp
#include <cuml/common/logger.hpp>

int main(int argc, char *argv[]) {
  CUML_LOG_WARN("This is a warning from the cuML logger!");
  return 0;
}
```

To compile this example, we must point the compiler to where cuML was
installed. Assuming you did not provide a custom `$CMAKE_INSTALL_PREFIX`, this
will default to the `$CONDA_PREFIX` environment variable.

```bash
$ export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
$ nvcc \
       main.cpp \
       -o cuml_logger_example \
       "-L${CONDA_PREFIX}/lib" \
       "-I${CONDA_PREFIX}/include" \
       "-I${CONDA_PREFIX}/include/cuml/raft" \
       -lcuml++
$ ./cuml_logger_example
[W] [13:26:43.503068] This is a warning from the cuML logger!
```
