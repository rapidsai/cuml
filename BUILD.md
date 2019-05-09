# cuML Build From Source Guide

## Setting Up Your Build Environment

To install cuML from source, ensure the dependencies are met:

1. [cuDF](https://github.com/rapidsai/cudf) (>=0.7)
2. zlib
3. cmake (>= 3.12.4)
4. CUDA (>= 9.2)
5. Cython (>= 0.29)
6. gcc (>=5.4.0)
7. BLAS - Any BLAS compatible with cmake's [FindBLAS](https://cmake.org/cmake/help/v3.12/module/FindBLAS.html). Note that the blas has to be installed to the same folder system as cmake, for example if using conda installed cmake, the blas implementation should also be installed in the conda environment.

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
$ mkdir build
$ cd build
$ export CUDA_BIN_PATH=$CUDA_HOME # (optional env variable if cuda binary is not in the PATH. Default CUDA_HOME=/path/to/cuda/)
$ cmake ..
```

If using a conda environment (recommended), then cmake can be configured appropriately via:

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

There are many options to configure the build process, see the [customizing build section](#custom-build-options).

3. Build `libcuml++`:

```bash
$ make -j
$ make install
```

To run tests (optional):

```bash
$ ./ml_test # Single GPU algorithm tests
$ ./ml_mg_test # Multi GPU algorithm tests
$ ./prims_test # ML Primitive function tests
```

If you want a list of the available tests:
```bash
$ ./ml_test --gtest_list_tests # Single GPU algorithm tests
$ ./ml_mg_test --gtest_list_tests # Multi GPU algorithm tests
$ ./prims_test --gtest_list_tests # ML Primitive function tests
```

4. Build the `cuml` python package:

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

### Custom Build Options

cuML's cmake has the following configurable flags available:


<sub>

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BLAS_LIBRARIES | path/to/blas_lib | "" | Optional variable allowing to manually specify location of BLAS library. |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. If either BUILD_CUML_TESTS or BUILD_CUML_MG_TESTS are set to ON, this variable is forced to be ON |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. |
| BUILD_PRIM_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |

</sub>


