# cuML Build From Source Guide

## Setting Up Your Build Environment

To install cuML from source, ensure the following dependencies are met:

1. [cuDF](https://github.com/rapidsai/cudf) (Same as cuML Version)
2. zlib
3. cmake (>= 3.26.4)
4. CUDA (>= 12+)
5. Cython (>= 0.29)
6. gcc (>= 13.0)
7. BLAS - Any BLAS compatible with cmake's [FindBLAS](https://cmake.org/cmake/help/v3.14/module/FindBLAS.html). Note that the blas has to be installed to the same folder system as cmake, for example if using conda installed cmake, the blas implementation should also be installed in the conda environment.
8. clang-format (= 20.1.4) - enforces uniform C++ coding style; required to build cuML from source. The packages `clang=20` and `clang-tools=20` from the conda-forge channel should be sufficient, if you are on conda. If not using conda, install the right version using your OS package manager.
9. NCCL (>=2.4)
10. UCX [optional] (>= 1.7) - enables point-to-point messaging in the cuML standard communicator. This is necessary for many multi-node multi-GPU cuML algorithms to function.

It is recommended to use conda for environment/package management. If doing so, development environment .yaml files are located in `conda/environments/all_*.yaml`. These files contains most of the dependencies mentioned above (notable exceptions are `gcc` and `zlib`). To create a development environment named `cuml_dev`, you can use the follow commands:

```bash
conda create -n cuml_dev python=3.13
conda env update -n cuml_dev --file=conda/environments/all_cuda-130_arch-x86_64.yaml
conda activate cuml_dev
```

## Installing from Source:

### Recommended process

As a convenience, a `build.sh` script is provided which can be used to execute the same build commands above.  Note that the libraries will be installed to the location set in `$INSTALL_PREFIX` if set (i.e. `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ ./build.sh                           # build the cuML libraries, tests, and python package, then
                                       # install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```
For workflows that involve frequent switching among branches or between debug and release builds, it is recommended that you install [ccache](https://ccache.dev/) and make use of it by passing the `--ccache` flag to `build.sh`.

To build individual components, specify them as arguments to `build.sh`
```bash
$ ./build.sh libcuml                   # build and install the cuML C++ and C-wrapper libraries
$ ./build.sh cuml                      # build and install the cuML python package
$ ./build.sh prims                     # build the ml-prims tests
$ ./build.sh bench                     # build the cuML c++ benchmark
$ ./build.sh prims-bench               # build the ml-prims c++ benchmark
```

Other `build.sh` options:
```bash
$ ./build.sh clean                     # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcuml -v                # build and install libcuml with verbose output
$ ./build.sh libcuml -g                # build and install libcuml for debug
$ PARALLEL_LEVEL=8 ./build.sh libcuml  # build and install libcuml limiting parallel build jobs to 8 (ninja -j8)
$ ./build.sh libcuml -n                # build libcuml but do not install
$ ./build.sh prims --allgpuarch        # build the ML prims tests for all supported GPU architectures
$ ./build.sh cuml --singlegpu          # build the cuML python package without MNMG algorithms
$ ./build.sh --ccache                  # use ccache to cache compilations, speeding up subsequent builds
```

By default, Ninja is used as the cmake generator. To override this and use (e.g.) `make`, define the `CMAKE_GENERATOR` environment variable accordingly:
```bash
CMAKE_GENERATOR='Unix Makefiles' ./build.sh
```

To run the C++ unit tests (optional), from the repo root:

```bash
$ cd cpp/build
$ ./test/ml # Single GPU algorithm tests
$ ./test/ml_mg # Multi GPU algorithm tests, if --singlegpu was not used
$ ./test/prims # ML Primitive function tests
```

If you want a list of the available C++ tests:
```bash
$ ./test/ml --gtest_list_tests # Single GPU algorithm tests
$ ./test/ml_mg --gtest_list_tests # Multi GPU algorithm tests
$ ./test/prims --gtest_list_tests # ML Primitive function tests
```


To run all Python tests, including multiGPU algorithms, from the repo root:
```bash
$ cd python
$ pytest -v
```

If only the single GPU algos want to be run, then:

```bash
$ pytest --ignore=cuml/tests/dask --ignore=cuml/tests/test_nccl.py
```

If you want a list of the available Python tests:
```bash
$ pytest cuML/tests --collect-only
```

Note: To run tests requiring `xgboost` in conda devcontainers, users must install the `xgboost` conda package manually.
See `dependencies.yaml` for more information.

### Manual Process

Once dependencies are present, follow the steps below:

1. Clone the repository.
```bash
$ git clone https://github.com/rapidsai/cuml.git
```

2. Build and install `libcuml++` (C++/CUDA library containing the cuML algorithms), starting from the repository root folder:
```bash
$ cd cpp
$ mkdir build && cd build
$ export CUDA_BIN_PATH=$CUDA_HOME # (optional env variable if cuda binary is not in the PATH. Default CUDA_HOME=/path/to/cuda/)
$ cmake ..
```

If using a conda environment (recommended), then cmake can be configured appropriately for `libcuml++` via:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```

Note: The following warning message is dependent upon the version of cmake and the `CMAKE_INSTALL_PREFIX` used. If this warning is displayed, the build should still run successfully. We are currently working to resolve this open issue. You can silence this warning by adding `-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib` to your `cmake` command.
```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```

The configuration script will print the BLAS found on the search path. If the version found does not match the version intended, use the flag `-DBLAS_LIBRARIES=/path/to/blas.so` with the `cmake` command to force your own version.

If using conda and a conda installed cmake, the `openblas` conda package is recommended and can be explicitly specified for `blas` and `lapack`:

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so
```

Additionally, to reduce compile times, you can specify a GPU compute capability to compile for, for example for Volta GPUs:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS="70"
```

You may also wish to make use of `ccache` to reduce build times when switching among branches or between debug and release builds:

```bash
$ cmake .. -DUSE_CCACHE=ON
```

There are many options to configure the build process, see the [customizing build section](#libcuml-&-libcumlc++).

3. Build `libcuml++` and `libcuml`:

```bash
$ make -j
$ make install
```

To run tests (optional):
```bash
$ ./test/ml # Single GPU algorithm tests
$ ./test/ml_mg # Multi GPU algorithm tests
$ ./test/prims # ML Primitive function tests
```

If you want a list of the available tests:
```bash
$ ./test/ml --gtest_list_tests # Single GPU algorithm tests
$ ./test/ml_mg --gtest_list_tests # Multi GPU algorithm tests
$ ./test/prims --gtest_list_tests # ML Primitive function tests
```

To run cuML c++ benchmarks (optional):
```bash
$ ./bench/sg_benchmark  # Single GPU benchmarks
```
Refer to `--help` option to know more on its usage

To run ml-prims C++ benchmarks (optional):
```bash
$ ./bench/prims_benchmark  # ml-prims benchmarks
```
Refer to `--help` option to know more on its uage

To build doxygen docs for all C/C++ source files
```bash
$ make doc
```

5. Build the `cuml` python package:

```bash
$ cd ../../python
$ python setup.py build_ext --inplace
```

To run Python tests (optional):

```bash
$ pytest -v
```


If only the single GPU algos want to be run, then:

```bash
$ pytest --ignore=cuml/tests/dask --ignore=cuml/tests/test_nccl.py
```


If you want a list of the available tests:
```bash
$ pytest cuML/tests --collect-only
```

5. Finally, install the Python package to your Python path:

```bash
$ python setup.py install
```

### Custom Build Options

#### libcuml & libcuml++

cuML's cmake has the following configurable flags available:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BLAS_LIBRARIES | path/to/blas_lib | "" | Optional variable allowing to manually specify location of BLAS library. |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_C_LIBRARY, BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_C_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml shared library. Setting this variable to `ON` will set the variable BUILD_CUML_CPP_LIBRARY to `ON` |
| BUILD_CUML_STD_COMMS | [ON, OFF] | ON | Enable/disable building cuML NCCL+UCX communicator for running multi-node multi-GPU algorithms. Note that UCX support can also be enabled/disabled (see below). Note that BUILD_CUML_STD_COMMS and BUILD_CUML_MPI_COMMS are not mutually exclusive and can both be installed simultaneously. |
| WITH_UCX | [ON, OFF] | OFF | Enable/disable UCX support for the standard cuML communicator. Algorithms requiring point-to-point messaging will not work when this is disabled. This has no effect on the MPI communicator. |
| BUILD_CUML_MPI_COMMS | [ON, OFF] | OFF | Enable/disable building cuML MPI+NCCL communicator for running multi-node multi-GPU C++ tests. Note that BUILD_CUML_STD_COMMS and BUILD_CUML_MPI_COMMS are not mutually exclusive, and can both be installed simultaneously. |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. |
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| BUILD_CUML_BENCH | [ON, OFF] | ON | Enable/disable building of cuML C++ benchark.  |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DETECT_CONDA_ENV | [ON, OFF] | ON | Use detection of conda environment for dependencies. If set to ON, and no value for CMAKE_INSTALL_PREFIX is passed, then it'll assign it to $CONDA_PREFIX (to install in the active environment).  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml++.  |
