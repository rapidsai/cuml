# cuML Build From Source Guide

## Setting Up Your Build Environment

To install cuML from source, ensure the following dependencies are met:

1. [cuDF](https://github.com/rapidsai/cudf) (>=0.8)
2. zlib
3. cmake (>= 3.14)
4. CUDA (>= 9.2)
5. Cython (>= 0.29)
6. gcc (>=5.4.0)
7. BLAS - Any BLAS compatible with cmake's [FindBLAS](https://cmake.org/cmake/help/v3.14/module/FindBLAS.html). Note that the blas has to be installed to the same folder system as cmake, for example if using conda installed cmake, the blas implementation should also be installed in the conda environment.
8. clang-format (= 8.0.0) - enforces uniform C++ coding style; required to build cuML from source. The RAPIDS conda channel provides a package. If not using conda, install using your OS package manager.
9. NCCL (>=2.4)

It is recommended to use conda for environment/package management. If doing so, a convenience environment .yml file is located in `conda/environments/cuml_dec_cudax.y.yml` (replace x.y for your CUDA version). This file contains most of the dependencies mentioned above (notable exceptions are `gcc` and `zlib`). To use it, for example to create an environment named `cuml_dev` for CUDA 10.0 and Python 3.7, you can use the follow command:

```
conda env create -n cuml_dev python=3.7 --file=conda/environments/cuml_dev_cuda10.0.yml
```

## Installing from Source:

### Typical Process

Once dependencies are present, follow the steps below:

1. Clone the repository.
```bash
$ git clone --recurse-submodules https://github.com/rapidsai/cuml.git
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

Note: The following warning message is dependent upon the version of cmake and the `CMAKE_INSTALL_PREFIX` used. If this warning is displayed, the build should still run succesfully. We are currently working to resolve this open issue. You can silence this warning by adding `-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib` to your `cmake` command.
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



4. Build and install `libcumlcomms` (C++/CUDA library enabling multi-node multi-GPU communications), starting from the repository root folder:
```bash
$ cd cpp/comms
$ mkdir build && cd build
$ cmake ..

```

If using a conda environment (recommended), then cmake can be configured appropriately for `libcumlcomms` via:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```


See the [customizing build section](#libcumlcomms) for options to configure the build process.


5. Build the `cuml` python package:

```bash
$ cd ../../python
$ python setup.py build_ext --inplace
```

To run Python tests (optional):

```bash
$ pytest -v
```

If you want a list of the available tests:
```bash
$ pytest cuML/test --collect-only
```

5. Finally, install the Python package to your Python path:

```bash
$ python setup.py install
```

#### `build.sh`

As a convenience, a `build.sh` script is provided which can be used to execute the same build commands above.  Note that the libraries will be installed to the location set in `$INSTALL_PREFIX` if set (i.e. `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ ./build.sh                           # build the cuML libraries, tests, and python package, then
                                       # install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```

To build individual components, specify them as arguments to `build.sh`
```bash
$ ./build.sh libcuml                   # build and install the cuML C++ and C-wrapper libraries
$ ./build.sh cuml                      # build and install the cuML python package
$ ./build.sh prims                     # build the ML prims tests
```

Other `build.sh` options:
```bash
$ ./build.sh clean                     # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcuml -v                # build and install libcuml with verbose output
$ ./build.sh libcuml -g                # build and install libcuml for debug
$ PARALLEL_LEVEL=4 ./build.sh libcuml  # build and install libcuml limiting parallel build jobs to 4 (make -j4)
$ ./build.sh libcuml -n                # build libcuml but do not install
$ ./build.sh prims --allgpuarch        # build the ML prims tests for all supported GPU architectures
$ ./build.sh cuml --multigpu           # build the cuml python package with multi-GPU support (requires libcumlMG and CUDA >= 10.0)
```

### Custom Build Options

#### libcuml & libcumlc++

cuML's cmake has the following configurable flags available:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BLAS_LIBRARIES | path/to/blas_lib | "" | Optional variable allowing to manually specify location of BLAS library. |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_C_LIBRARY, BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_C_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml shared library. Setting this variable to `ON` will set the variable BUILD_CUML_CPP_LIBRARY to `ON` |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. |
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml++.  |


#### libcumlcomms

cuML's multi-GPU communicator cmake has the following configurable flags available:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| WITH_UCX | [ON, OFF]  | OFF  | Enable/disable point-to-point support with UCX (experimental) |
| CUML_INSTALL_DIR | /path/to/libcuml++.so | "" | Specifies location of libcuml for linking |

