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

### Installing from Source:

Once dependencies are present, follow the steps below:

1. Clone the repository.
```bash
$ git clone --recurse-submodules https://github.com/rapidsai/cuml.git
```

2. Build and install `libcuml` (the C++/CUDA library containing the cuML algorithms), starting from the repository root folder:
```bash
$ cd cuML
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

Additionally, to reduce compile times, you can specify a GPU compute capability to compile for, for example for Volta GPUs:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS="70"
```


3. Build `libcuml`:

```bash
$ make -j
$ make install
```

To run tests (optional):

```bash
$ ./ml_test
```

If you want a list of the available tests:
```bash
$ ./ml_test --gtest_list_tests
```

4. Build the `cuml` python package:

```bash
$ cd ../../python
$ python setup.py build_ext --inplace
```

To run Python tests (optional):

```bash
$ py.test -v
```

If you want a list of the available tests:
```bash
$ py.test cuML/test --collect-only
```

5. Finally, install the Python package to your Python path:

```bash
$ python setup.py install
```

6. You can also build and run tests for the machine learning primitive header only library located in the `ml-prims` folder. From the repository root:

```bash
$ cd ml-prims
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS="70" # specifying GPU_ARCH is optional, but significantly reduces compile time
$ make -j
```

To run the ml-prim tests:

```bash
$./test/mlcommon_test
```


