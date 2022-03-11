# cuML Build From Source Guide



## Fast Guide

## Setting Up Your Build Environment

There are two groups of dependencies that need to be met, the core dependencies to build the C++ artifacts, and then the dependencies of the Python package which are a superset of those.

There are predefined conda environments that meet all the requirements, or they can be added manually. Additionally, many requirements can be fetched automatically by CMake (using CPM), but that can lead to significantly slower build times.

### Conda Developer Environments

If you are using conda, you can find 3 types of pre-defined environments:

- `libcuml_dev_cuda11.5.yml`: Creates a conda environment suitable to build the C++ artifacts.
- `cuml_dev_cuda11.5.yml`: Creates a conda environment suitable to build the C++ and Python artifacts.
- `rapids_dev_cuda11.5yml`: Creates a conda environment suitable to build any RAPIDS project, including cuML, cuDF and cuGraph.

If you require another 11.x version of CUDA, just edit the `cuatoolkit=11.5` line inside those files. Note that cuDF *requires* CUDA>=11.5 to be built, so take that into consideration if you are using the `rapids_dev_cuda11.5yml` to compile cuDF.
It is recommended to use `mamba`() to speed up creating the environments, , but you can use `conda` as well:

```bash
mamba env create -f conda/environments/libcuml_dev_cuda11.5.yml python=3.9 -n libcuml_dev
```

If you're using the `rapids_dev_cuda11.5yml` environment that can build all of RAPIDS and want to upgrade any of the packages in it, you must first remove the meta-packages in it with:

```bash
conda remove --force rapids-build-env rapids-notebook-env rapids-doc-env
```


### Docker Developer Container

The recommended way to use docker for development is to use RAPIDS-compose https://github.com/trxcllnt/rapids-compose

### C++ Dependencies

To build `libcuml++`, `libcuml` and related components, the following dependencies are needed:

1. `CUDA` >= 11.0, 11.5 recommended.
2. `GCC`/`G++` >= 9.3
3. `CMake` >= 3.20.1
4. `ninja`
5. Optional: `sccache` or `ccache` to speedup re-compilations.
6. `RMM` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04). If not found, it will be fetched by CMake.
7. `libraft-headers`=22.04.* If not found, it will be fetched by CMake.
8. `libraft-distance`=22.04.* If not found, it will be fetched by CMake. Using the precompiled binaries from the conda packages speeds up compilation significantly.
9. `libraft-nn`=22.04.* If not found, it will be fetched by CMake. Using the precompiled binaries from the conda packages speeds up compilation significantly.
10. `treelite`=2.3.0 If not found, it will be fetched by CMake.
11. `libcumlprims` for multiGPU C++ algorithms (Read section on multigpu components).
12. `UCX` with CUDA support >=1.7 for multiGPU C++ algorithms (Read section on multigpu components).
13. `NCCL` (>=2.4) for multiGPU C++ algorithms (Read section on multigpu components).
14. Optional `doxygen` >=1.8.20 for generating documentation


### Python Dependencies

To build the `cuml` Python package, the C++ requirements are needed plus:

15. `cuda-python` (corresponding to the CUDA version of the system)
16. `cuDF` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).
17. `pyraft` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).
18. `dask-cudf` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).
19. `dask-cuda` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).


### Python Unit Test Dependencies

To run the (`pytest` based)

20. `pytest`
21. `scikit-learn=0.24`
22. `dask-ml`
23. `umap-learn`
24. `statsmodels`
25. `seaborn`
26. `hdbscan`
27. `nltk`

## Build Process

### Fast Process

As a convenience, a `build.sh` script is provided which can be used to execute the necessary CMake build commands automatically with a fair degree of configuration.  The libraries will be installed to the location set in `$INSTALL_PREFIX` if set (i.e. `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`:

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

By default, `Ninja` is used as the cmake generator. To override this and use (e.g.) `make`, define the `CMAKE_GENERATOR` environment variable accodingly:
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
$ pytest --ignore=cuml/test/dask --ignore=cuml/test/test_nccl.py
```

If you want a list of the available Python tests:
```bash
$ pytest cuML/test --collect-only
```

### Full Process

Once dependencies are present, to build and install `libcuml++` (C++/CUDA library containing the cuML algorithms), starting from the repository root folder:

```bash
$ cd cpp
$ mkdir build && cd build
$ export CUDA_BIN_PATH=$CUDA_HOME # (optional env variable if cuda binary is not in the PATH. Default CUDA_HOME=/path/to/cuda/)
$ cmake ..
```

Note: The following warning message is dependent upon the version of cmake and the `CMAKE_INSTALL_PREFIX` used. If this warning is displayed, the build should still run succesfully.

```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```


Additionally, to reduce compile times, you can specify a GPU compute capability to compile for, for example for the system's GPU architecture:

```bash
$ cmake .. -DGPU_ARCHS=NATIVE
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
$ pytest --ignore=cuml/test/dask --ignore=cuml/test/test_nccl.py
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
| BUILD_CUML_PRIMS_BENCH | [ON, OFF] | ON | Enable/disable building of ml-prims C++ benchark.  |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DETECT_CONDA_ENV | [ON, OFF] | ON | Use detection of conda environment for dependencies. If set to ON, and no value for CMAKE_INSTALL_PREFIX is passed, then it'll assign it to $CONDA_PREFIX (to install in the active environment).  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml++.  |
