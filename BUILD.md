### Dependencies for Installing/Building from Source:

To install cuML from source, ensure the dependencies are met:

1. [cuDF](https://github.com/rapidsai/cudf) (>=0.5.1)
2. zlib Provided by zlib1g-dev in Ubuntu 16.04
3. cmake (>= 3.12.4)
4. CUDA (>= 9.2)
5. Cython (>= 0.29)
6. gcc (>=5.4.0)
7. BLAS - Any BLAS compatible with Cmake's [FindBLAS](https://cmake.org/cmake/help/v3.12/module/FindBLAS.html)

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

If using a conda environment (recommended currently), then cmake can be configured appropriately via:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```

Note: The following warning message is dependent upon the version of cmake and the `CMAKE_INSTALL_PREFIX` used. If this warning is displayed, the build should still run succesfully. We are currently working to resolve this open issue. You can silence this warning by adding `-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib` to your `cmake` command.
```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```

The configuration script will print the BLAS found on the search path. If the version found does not match the version intended, use the flag `-DBLAS_LIBRARIES=/path/to/blas.so` with the `cmake` command to force your own version.


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

cuML's core structure contains:

1. ***cuML***:
  C++/CUDA machine learning algorithms. This library currently includes the following six algorithms:
  - Single GPU Truncated Singular Value Decomposition (tSVD)
  - Single GPU Principal Component Analysis (PCA)
  - Single GPU Density-based Spatial Clustering of Applications with Noise (DBSCAN)
  - Single GPU Kalman Filtering
  - Multi-GPU K-Means Clustering
  - Multi-GPU K-Nearest Neighbors (Uses [Faiss](https://github.com/facebookresearch/faiss))

2. ***python***:
  Python bindings for the above algorithms, including interfaces for [cuDF](https://github.com/rapidsai/cudf). These bindings connect the data to C++/CUDA based cuML and ml-prims libraries without leaving GPU memory.

3. ***ml-prims***:
  Low level machine learning primitives header only library, used in cuML algorithms. Includes:
  - Linear Algebra
  - Statistics
  - Basic Matrix Operations
  - Distance Functions
  - Random Number Generation

## External

The external folders contains submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main `CMakeLists.txt` file for these.

Current external submodules are:

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Google Test](https://github.com/google/googletest)
- [CUB](https://github.com/NVlabs/cub)
