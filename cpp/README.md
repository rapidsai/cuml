# cuML C++

This folder contains the C++ and CUDA code of the algorithms and ML primitives of cuML. The build system uses CMake for build configuration, and an out-of-source build is recommended.

## Source Code Folders

The source code of cuML is divided in three main directories: `src`, `src_prims`, and `comms`.

- `src` contains the source code of the Machine Learning algorithms, and the main cuML C++ API. The main consumable is the shared library `libcuml++`, that can be used stand alone by C++ consumers or is consumed by our Python package `cuml` to provide a Python API.
- `src_prims` contains most of the common components and computational primitives that form part of the machine learning algorithms in cuML, and can be used individually as well in the form of a header only library.
- `comms` contains the source code of the communications implementations that enable multi-node multi-GPU algorithms. There are currently two communications implementations. The implementation in the `mpi` directory is for MPI environments. It can also be used for automated tested. The implementation in the `std` directory is required for running cuML in multi-node multi-GPU Dask environments.

The `test` directory has subdirectories that reflect this distinction between the `src` and `prims` components of cuML.

## Setup
### Dependencies

1. cmake (>= 3.26.4)
2. CUDA (>= 12.0)
3. gcc (>=13.0)
4. clang-format (= 20.1.4) - enforces uniform C++ coding style; required to build cuML from source. The packages `clang=20` and `clang-tools=20` from the conda-forge channel should be sufficient, if you are on conda. If not using conda, install the right version using your OS package manager.

### Building cuML:

The main artifact produced by the build system is the shared library libcuml++. Additionally, executables to run tests for the algorithms can be built. To see detailed steps see the [BUILD](../BUILD.md) document of the repository.

Current cmake offers the following configuration options:

- Build Configuration Options:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_C_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. Requires MPI to be installed. When enabled, BUILD_CUML_MPI_COMMS will be automatically set to ON. See section about additional requirements.|
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| BUILD_CUML_BENCH | [ON, OFF]  | ON  | Enable/disable building of cuML C++ benchark. |
| BUILD_CUML_STD_COMMS | [ON, OFF] | ON | Enable/disable building cuML NCCL+UCX communicator for running multi-node multi-GPU algorithms. Note that UCX support can also be enabled/disabled (see below). The standard communicator and MPI communicator are not mutually exclusive and can both be installed at the same time. |
| WITH_UCX | [ON, OFF] | OFF | Enable/disable UCX support in the standard cuML communicator. Algorithms requiring point-to-point messaging will not work when this is disabled. This flag is ignored if BUILD_CUML_STD_COMMS is set to OFF. |
| BUILD_CUML_MPI_COMMS | [ON, OFF] | OFF | Enable/disable building cuML MPI+NCCL communicator for running multi-node multi-GPU C++ tests. MPI communicator and STD communicator may both be installed at the same time. If OFF, it overrides BUILD_CUML_MG_TESTS to be OFF as well. |
| SINGLEGPU | [ON, OFF] | OFF | Disable all mnmg components. Disables building of all multi-GPU algorithms and all comms library components. Removes libcumlprims, UCXX and NCCL dependencies. Overrides values of  BUILD_CUML_MG_TESTS, BUILD_CUML_STD_COMMS, WITH_UCX and BUILD_CUML_MPI_COMMS. |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| CMAKE_CUDA_ARCHITECTURES |  List of GPU architectures, semicolon-separated | Empty  | List the GPU architectures to compile the GPU targets for. Set to "NATIVE" to auto detect GPU architecture of the system, set to "ALL" to compile for all RAPIDS supported archs: ["60" "62" "70" "72" "75" "80" "86"].  |
| USE_CCACHE | [ON, OFF]  | ON  | Cache build artifacts with ccache. |

- Debug configuration options:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml++.  |

After running CMake in a `build` directory, if the `BUILD_*` options were not turned `OFF`, the following targets can be built:

```bash
$ cmake --build . -j                        # Build libcuml++ and all tests
$ cmake --build . -j --target  sg_benchmark # Build c++ cuml single gpu benchmark
$ cmake --build . -j --target  cuml++       # Build libcuml++
$ cmake --build . -j --target  ml           # Build ml_test algorithm tests binary
$ cmake --build . -j --target  ml_mg        # Build ml_mg_test multi GPU algorithms tests binary
$ cmake --build . -j --target  prims        # Build prims_test ML primitive unit tests binary
```

### MultiGPU Tests Requirements Note:

To build the MultiGPU tests (CMake option `BUILD_CUML_MG_TESTS`), the following dependencies are required:

- MPI (OpenMPI recommended)
- NCCL, version corresponding to [RAFT's requirement](https://github.com/rapidsai/raft/blob/branch-23.02/conda/recipes/raft-dask/meta.yaml#L49.

### Third Party Modules

The external folder contains submodules that cuML depends on.

Current external submodules are:

1. [CUB](https://github.com/NVlabs/cub)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [Google Test](https://github.com/google/googletest)

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
