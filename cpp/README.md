# cuML C++

This folder contains the C++ and CUDA code of the algorithms and ML primitives of cuML. The build system uses CMake for build configuration, and an out-of-source build is recommended.

## Source Code Folders

The source code of cuML is divided in two main files: `src` and `src_prims`.

- `src` contains the source code of the Machine Learning algorithms, and the main cuML C++ API. The main consumable is the shared library `libcuml++`, that can be used stand alone by C++ consumers or is consumed by our Python package `cuml` to provide a Python API.
- `src_prims` contains most of the common components and computational primitives that form part of the machine learning algorithms in cuML, and can be used individually as well in the form of a header only library.

The test folder has subfolders that reflect this distinction between the components of cuML.

## Setup
### Dependencies

1. zlib
2. cmake (>= 3.12.4)
3. CUDA (>= 9.2)
4. gcc (>=5.4.0)
5. BLAS - Any BLAS compatible with cmake's [FindBLAS](https://cmake.org/cmake/help/v3.12/module/FindBLAS.html). Note that the blas has to be installed to the same folder system as cmake, for example if using conda installed cmake, the blas implementation should also be installed in the conda environment.
6. clang-format (= 8.0.0) - This is needed if you are building this C++ library from source. For the ones using conda environment, our rapids official channel should have this available, until then, the installation command is `conda install -c teju85 libclang`. For the ones not using conda, please install this version using your favorite packaging SW (viz., yum, apt-get, ...). This is needed to enforce uniform C++ coding style across the project.

### Building cuML:

The main artifact produced by the build system is the shared library libcuml++. Additionally, executables to run tests for the algorithms can be built. To see detailed steps see the [BUILD](../BUILD.md) document of the repository.

Current cmake offers the following configuration options:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BLAS_LIBRARIES | path/to/blas_lib | "" | Optional variable allowing to manually specify location of BLAS library. |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. |
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |

After running CMake in a `build` directory, if the `BUILD_*` options were not turned `OFF`, the following targets can be built:

```bash
$ make -j # Build libcuml++ and all tests
$ make -j cuml++ # Build libcuml++
$ make -j ml # Build ml_test algorithm tests binary
$ make -j ml_mg # Build ml_mg_test multi GPU algorithms tests binary
$ make -j prims # Build prims_test ML primitive unit tests binary
```

### Third Party Modules

The external folder contains submodules that cuML depends on.

Current external submodules are:

1. [CUTLASS](https://github.com/NVIDIA/cutlass)
2. [CUB](https://github.com/NVlabs/cub)
3. [Faiss](https://github.com/facebookresearch/faiss)
4. [Google Test](https://github.com/google/googletest)
