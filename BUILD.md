# cuML Build From Source Guide

## Table of Contents

- [Quick Start Guide](#quick-start-guide)
- [Setting Up Your Build Environment](#setting-up-your-build-environment)
    - [Conda Developer Environments](#conda-developer-environments)
    - [Note About Using Docker](#docker-developer-container)
    - [C++ Dependencies](#c-dependencies)
    - [Python Dependencies](#python-dependencies)
    - [Python Unit Test Dependencies](#python-unit-test-dependencies)
- [Build Guide](#build-guide)
    - [Using build.sh](#using-buildsh-for-c-and-python)
    - [Building C++ Using CMake](#building-c-artifacts-with-cmake)
        - [Single and Multi-GPU Components](#single-and-multi-gpu-components)
        - [Configuring Algorithms Built](#configuring-algorithms-built)
    - [Building Python using `setup.py`](#building-python-artifacts-with-setuppy)
- [Unit Tests](#unit-tests)
    - [C++ Unit Tests](#c-unit-tests)
    - [Python Unit Tests](#python-unit-tests)

## Quick Start Guide

1. System must have the following minimum requirements:

- Pascal or newer NVIDIA GPU
- `CUDA` 11.0 or greater. You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- `CMake` 3.20.1 or greater
- `GCC`/`G++` 9.3 or greater
- `Ninja` or `make`
- `OpenMP`
- `conda` or `mamba` (`mamba` is recommended to speed up the process.) **Note:** The single GPU C++ artifacts can be built without any further dependencies and without `conda`/`mamba` and the corresponding developer environments, though at the cost of _significantly_ longer compile time and binary size. For more details see [Building C++ Using CMake](#building-c-artifacts-with-cmake).

2. Choose and create the appropriate conda development environment:

- For building the C++ library only:
```bash
mamba env create -f conda/environments/libcuml_dev_cuda11.5.yml -n libcuml_dev # change libcuml_dev for any name you want for the environment
```

- For building either the C++ or the Python libraries or both:
```bash
mamba env create -f conda/environments/cuml_dev_cuda11.5.yml python=3.9 -n cuml_dev # change libcuml_dev for any name you want for the environment
```

3. From the root folder of the repository, use the convenience script build.sh to build the appropriate artifacts:

- For building the C++ library only:
```bash
PARALLEL_LEVEL=8 ./build.sh libcuml
```

By default, `Ninja` is used as the cmake generator. To override this and use (e.g.) `make`, define the `CMAKE_GENERATOR` environment variable accordingly:
```bash
PARALLEL_LEVEL=8 CMAKE_GENERATOR='Unix Makefiles' ./build.sh
```

- For building either the C++ or the Python libraries (or both):
```bash
PARALLEL_LEVEL=8 ./build.sh libcuml cuml
```

In all of the above commands, you can increase the parallel level to speed up compilation with more parallelism at the cost of needing more RAM in the system.

4. Run unit tests if desired:

- C++ tests (from the repository root folder):

```bash
cd cpp/build
ninja test
```

- Python tests (from the repository root folder):

```bash
cd python
pytest cuml/test
````


## Setting Up Your Build Environment

There are two groups of dependencies that need to be met, the core dependencies to build the C++ artifacts, and then the dependencies of the Python package which are a superset of those.

There are predefined conda environments that can be used to install all the requirements, or they can be added manually. Additionally, many requirements can be fetched automatically by CMake (using CPM), but that can lead to significantly slower build times.

### Conda Developer Environments

If you are using conda, you can find 3 types of pre-defined environments:

- `libcuml_dev_cuda11.5.yml`: Creates a conda environment suitable to build the C++ artifacts.
- `cuml_dev_cuda11.5.yml`: Creates a conda environment suitable to build the C++ and Python artifacts.
- `rapids_dev_cuda11.5yml`: Creates a conda environment suitable to build any RAPIDS project, including cuML, cuDF and cuGraph. Note, it doesn't include those packages, so if you want to build cuML Python in it, you can build or install cuDF in it.

If you require another 11.x version of CUDA, just edit the `cudatoolkit=11.5` line inside those files. **Note**: cuDF requires CUDA>=11.5 to be built, so take that into consideration if you are using the `rapids_dev_cuda11.5yml` to compile cuDF.

It is recommended to use [`mamba`](https://mamba.readthedocs.io/en/latest/) to speed up creating the environments, but you can use `conda` as well:

```bash
mamba env create -f conda/environments/libcuml_dev_cuda11.5.yml python=3.9 -n libcuml_dev # change libcuml_dev for any name you want for the environment
```

If you already have an environment with the `libcuml++` dependencies, say named `libcuml_dev`, and want to add the Python dependencies, that can be done with:

```bash
mamba env update --file conda/environments/cuml_dev_cuda11.5.yml --name libcuml_dev
```

If a significant amount of time has passed between the creation of your environmente, there might be significant conflicts that could make this update fail, in that case it is suggested to create new environments.

**Note**: If you're using the `rapids_dev_cuda11.5yml` environment that can build all of RAPIDS and want to upgrade any of the packages in it, you must first remove the meta-packages in it with:

```bash
conda remove --force rapids-build-env rapids-notebook-env rapids-doc-env
```

### Docker Developer Container

To use Docker containers for development, there are 2 options, one for building already released RAPIDS versions, and another one that can be used for development verions:

- For released versions, the containers `rapidsai/rapidsai-dev:22.02-cuda11.5-devel-ubuntu20.04-py3.9` (replacing for the appropiate versions of RAPIDS, Ubuntu and Python) can be used. **Note** This are released as part of the general RAPIDS released version, so they are only apt to build the already released cuML version.

- For current, in development versions, the recommended way to use docker for development is to use [RAPIDS-compose](https://github.com/trxcllnt/rapids-compose).

### C++ Dependencies

To build `libcuml++`, `libcuml` and related components, the following dependencies are needed:

1. `CUDA` >= 11.0, 11.5 recommended.
2. `GCC`/`G++` >= 9.3
3. `CMake` >= 3.20.1
4. `ninja`
5. Optional: `sccache` or `ccache` to speedup re-compilations.
6. `RMM` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04). If not found, it will be fetched by CMake.
7. `libraft-headers` corresponding to the branch/version being built. If not found, it will be fetched by CMake.
8. `libraft-distance` corresponding to the branch/version being built. If not found, it will be fetched by CMake. Using the precompiled binaries from the conda packages speeds up compilation significantly.
9. `libraft-nn` corresponding to the branch/version being built. If not found, it will be fetched by CMake. Using the precompiled binaries from the conda packages speeds up compilation significantly.
10. `treelite`=2.3.0 If not found, it will be fetched by CMake.
11. `libcumlprims` for multiGPU C++ algorithms (Read section on multigpu components).
12. `UCX` with CUDA support >=1.7 for multiGPU C++ algorithms (Read section on multigpu components).
13. `NCCL` (>=2.4) for multiGPU C++ algorithms (Read section on multigpu components).
14. Optional `doxygen` >=1.8.20 for generating documentation


### Python Dependencies

To build the `cuml` Python package, the C++ requirements are needed plus:

15. `cuda-python` corresponding to the CUDA version of the system/environment.
16. `cuDF` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).
17. `pyraft` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).
18. `dask-cudf` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).
19. `dask-cuda` corresponding to the branch/version being built (i.e. 22.04 for branch-22.04).


### Python Unit Test Dependencies

To run the (`pytest` based) Python unit tests, the following additional packages are needed:

20. `pytest`
21. `scikit-learn=0.24`
22. `dask-ml`
23. `umap-learn`
24. `statsmodels`
25. `seaborn`
26. `hdbscan`
27. `nltk`

## Build Guide

### Using `build.sh` for C++ and Python

As a convenience, a `build.sh` script is provided which can be used to execute the necessary CMake build commands automatically with a fair degree of configuration.  The libraries will be installed to the location set in `$INSTALL_PREFIX` if set (i.e. `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`:

```bash
$ PARALLEL_LEVEL=8 ./build.sh          # build the cuML libraries, tests, and python package, then
                                       # install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```
For workflows that involve frequent switching among branches or between debug and release builds, it is recommended that you install either [ccache](https://ccache.dev/) or [sccache](https://docs.rs/crate/sccache/0.2.5) and make use of them by using the `--cachetool` flag to `build.sh`.

To build individual components, specify them as arguments to `build.sh`
```bash
$ PARALLEL_LEVEL=8 ./build.sh libcuml                   # build and install the cuML C++ and C-wrapper libraries
$ PARALLEL_LEVEL=8 ./build.sh libcuml cuml              # build and install the cuML C++ and Python packages
```

The complete list of `build.sh` options and flags:

```bash
./build.sh [<target> ...] [<flag> ...]
 where <target> is:
   clean             - remove all existing build artifacts and configuration (start over)
   libcuml           - build the libcuml++.so C++ library
   libcuml_c         - build the libcuml.so C library containing C wrappers around libcuml++.so
   cuml              - build the cuml Python package
   cppmgtests        - build libcuml++ mnmg tests. Builds MPI communicator, adding MPI as dependency.
   cppexamples       - build libcuml++ examples.
   prims             - build the ml-prims tests
   bench             - build the libcuml C++ benchmark
   prims-bench       - build the ml-prims C++ benchmark
   cppdocs           - build the C++ API doxygen documentation
   pydocs            - build the general and Python API documentation
 and <flag> is:
   -v                - verbose build mode
   -g                - build for debug
   -n                - no install step
   -h                - print this text
   --allgpuarch      - build for all supported GPU architectures
   --singlegpu       - Build libcuml and cuml without multigpu components
   --nolibcumltest   - disable building libcuml C++ tests for a faster build
   --nvtx            - Enable nvtx for profiling support
   --show_depr_warn  - show cmake deprecation warnings
   --codecov         - Enable code coverage support by compiling with Cython linetracing
                       and profiling enabled (WARNING: Impacts performance)
   --ccache          - Use ccache to speed up rebuilds. Deprecated, use '--cachectool ccache' insted.
   --cachetool:      - Specify one of sccache | ccache for speeding up builds and rebuilds.
   --nocloneraft     - CMake will clone RAFT even if it is in the environment, use this flag to disable that behavior
   --static-faiss    - Force CMake to use the FAISS static libs, cloning and building them if necessary
   --static-treelite - Force CMake to use the Treelite static libs, cloning and building them if necessary

 default action (no args) is to build and install 'libcuml', 'cuml', and 'prims' targets only for the detected GPU arch

 The following environment variables are also accepted to allow further customization:
   PARALLEL_LEVEL         - Number of parallel threads to use in compilation.
   CUML_EXTRA_CMAKE_ARGS  - Extra arguments to pass directly to cmake. Values listed in environment
                            variable will override existing arguments. Example:
                            CUML_EXTRA_CMAKE_ARGS="-DBUILD_CUML_C_LIBRARY=OFF" ./build.sh
   CUML_EXTRA_PYTHON_ARGS - Extra argument to pass directly to python setup.py
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

### Building C++ Artifacts with CMake

Once dependencies are present, to build and install `libcuml++` (C++/CUDA library containing the cuML algorithms), starting from the repository root folder:

```bash
$ cd cpp
$ mkdir build && cd build
$ export CUDA_BIN_PATH=$CUDA_HOME # (optional env variable if cuda binary is not in the PATH. Default CUDA_HOME=/path/to/cuda/)
$ cmake ..
```

**Note**: The following warning message is dependent upon the version of cmake and the `CMAKE_INSTALL_PREFIX` used. If this warning is displayed, the build should still run succesfully.

```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```

Additionally, to reduce compile times, you can specify a GPU compute capability to compile for, for example for the system's GPU architecture:

```bash
$ cmake .. -DGPU_ARCHS=NATIVE
```

You may also wish to make use of `ccache` or `sccache` to reduce build times when switching among branches or between debug and release builds by setting the appropriate flags, for example for `ccache`:

```bash
$ cmake .. -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

After CMake has run, you can build the C++ targets:

```bash
$ ninja -j  # replace with make if you're not using Ninja
$ ninja install
```

To build doxygen docs for all C/C++ source files
```bash
$ ninja doc
```

#### Single and Multi-GPU Components

By default, `libcuml++` will be built with all the algorithms in the codebase, including single and multi-GPU components. The multi-GPU algorithms add the following dependencies:

- `libcumlprims`: Closed-source library, the algorithms that depend on it include: linear models, `PCA`, and `tSVD`.
- `UCX`
- `NCCL`

`libcuml++` can be configured to avoid these dependencies, with the following CMake options:

- `-DENABLE_CUMLPRIMS_MG=OFF`: Disables algorithms that depend on `libcumlprims`. Still depends on `UCX` and `NCCL` to run the remaining multi-GPU algorithms.
- `-DSINGLE_GPU=ON`: Disables all multi-GPU algorithms, avoiding all `UCX`, `NCCL` and `libcumlprims` dependencies.

Using the `SINGLE_GPU` option allows an "all-from-source" build, where the only dependencies are the ones in point 1. of the [Quick Start Guide](#quick-start-guide). That said, depending on your intentions, particularly to create a single binary that statically includes all dependencies, the following options will be useful:

-`-DCUML_USE_RAFT_STATIC=ON` to statically link all RAFT binary dependencies.
-`-DCUML_USE_FAISS_STATIC=ON` to statically link FAISS.
-`-DCUML_USE_TREELITE_STATIC=ON` to statically link Treelite.

#### Configuring Algorithms Built

By default the `libcuml++.so` produced includes all single and multi-GPU algorithms, or only the single-GPU algorithms is configured that way as explained in the prior section. But sometimes further configurability is required, where `libcuml++.so` with only certain algorithms is desired. The build system has an experimental feature that allows this by using the following CMake option:

- `-DCUML_ALGORITHMS`: This CMake option allows to specify exactly which groups of algorithms, or individual algorithms, will be built into `libcuml++.so`.

**Note:** This is still an experimental/beta, in-progress, option of the build system, so no guarantees of everything working perfectly are set yet. Particularly, it doesn't support building the C-wrapper library (`libcuml`) and has not been extensively tested yet.

Specifying a single or set of algorithms can reduce compilation time, binary size as well as reduce the dependencies needed at built and runtime. Strings (with any casing, but uppercase is recommended for consistency with other CMake options) and semicolon-separated list of strings are accepted, where each string can be a single algorithms or group of algorithms. Possible options include:

- `"ALL"` is the default option, builds all algorithms into `libcuml++.so`.
- `"CLUSTER"` to include all clustering algorithms, or specify the individual options: `"DBSCAN"`, `"HDBSCAN"`, `"KMEANS"`, `"HIERARCHICALCLUSTERING"`, `"SPECTRALCLUSTERING"`.
- `"DECOMPOSITION"` to include decomposition algorithms, or specify indivual options: `"PCA"`, `"TSVD"`
- `"ENSEMBLE"` or `"RANDOMFOREST"` to include Random Forest (and Decision Tree) algorithms.
- `"FIL"` to include the Forest Inferencing Library (FIL).
- `"KNN"` to include the Nearest Neighbors models.
- `"LINEAR_MODEL"` to include linear models, or specify indivual options: `"LINEARREGRESSION"`, `"RIDGE"`, `"LASSO"`, `"LOGISTICREGRESSION"`.
- `"MANIFOLD"` to include manifold models, or specify indivual options: `"TSNE"`, `"UMAP"`.
- `"METRICS` to include metrics/scoring algorithms.
- `"SOLVERS"` to include manifold models, or specify indivual options: `"LARS", `"CD", `"SGD", `"QN"
- `"TSA"` to include time series models, or specify individual options: `"ARIMA"`, `"AUTOARIMA"`, `"HOLTWINTERS"`
- `"TREESHAP"` to include GPUTreeSHAP.

MultiGPU algorithms as well as any other missing components will be added in future releases.

#### Full Build Configuration Options

There are many options to configure the build process with the following CMake options:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_C_LIBRARY, BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_C_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml shared library. Setting this variable to `ON` will set the variable BUILD_CUML_CPP_LIBRARY to `ON` |
| BUILD_CUML_STD_COMMS | [ON, OFF] | ON | Enable/disable building cuML NCCL+UCX communicator for running multi-node multi-GPU algorithms. Note that UCX support can also be enabled/disabled (see below). Note that BUILD_CUML_STD_COMMS and BUILD_CUML_MPI_COMMS are not mutually exclusive and can both be installed simultaneously. |
| BUILD_CUML_MPI_COMMS | [ON, OFF] | OFF | Enable/disable building cuML MPI+NCCL communicator for running multi-node multi-GPU C++ tests. Note that BUILD_CUML_STD_COMMS and BUILD_CUML_MPI_COMMS are not mutually exclusive, and can both be installed simultaneously. |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. |
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| BUILD_CUML_BENCH | [ON, OFF] | ON | Enable/disable building of cuML C++ benchark.  |
| BUILD_CUML_PRIMS_BENCH | [ON, OFF] | ON | Enable/disable building of ml-prims C++ benchark.  |
| DETECT_CONDA_ENV | [ON, OFF] | ON | Use detection of conda environment for dependencies. If set to ON, and no value for CMAKE_INSTALL_PREFIX is passed, then it'll assign it to $CONDA_PREFIX (to install in the active environment).  |
| GPU_ARCHS |  List of GPU architectures, semicolon-separated | 60;70;75  | List of GPU architectures that all artifacts are compiled for.  |
| CUDA_ENABLE_KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| CUDA_ENABLE_LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml++.  |
| ENABLE_CUMLPRIMS_MG | [ON, OFF]  | ON  | Enable algorithms that use libcumlprims_mg.  |
| CUML_USE_FAISS_STATIC | [ON, OFF]  | OFF  | Build and statically link the FAISS library for nearest neighbors search on GPU.  |
| CUML_USE_TREELITE_STATIC | [ON, OFF]  | OFF  | Build and statically link the treelite library.  |


### Building Python Artifacts with `setup.py`

To build the `cuml` python package, from the repository root directory:

```bash
$ cd python
$ python setup.py build_ext --inplace
```

Afterwards, install the Python package to your Python path:

```bash
$ python setup.py install
```


## Unit Tests

### C++ Unit Tests

To run all the C++ unit tests, from the repo root after building the C++ artifacts:

```bash
$ cd cpp/build
$ ninja test
```

If you want a list of the available C++ tests, you can see them in the folder `test`, so from the `build` folder:

```bash
$ cd test
$ ls -lt
total 31100
-rwxrwxr-x  1 galahad galahad 6725408 Mar 15 20:25 SG_TSNE_TEST
-rwxrwxr-x  1 galahad galahad 6179888 Mar 15 20:25 SG_UMAP_PARAMETRIZABLE_TEST
-rwxrwxr-x  1 galahad galahad 3135832 Mar 15 20:25 SG_SVC_TEST
-rwxrwxr-x  1 galahad galahad  454192 Mar 15 20:25 SG_QUASI_NEWTON
-rwxrwxr-x  1 galahad galahad 1553280 Mar 15 20:25 SG_RF_TEST
-rwxrwxr-x  1 galahad galahad 2727696 Mar 15 20:25 SG_HDBSCAN_TEST
-rwxrwxr-x  1 galahad galahad  603800 Mar 15 20:25 SG_FIL_TEST
-rwxrwxr-x  1 galahad galahad  339776 Mar 15 20:25 SG_KNN_TEST
-rwxrwxr-x  1 galahad galahad  591664 Mar 15 20:25 SG_TSVD_TEST
-rwxrwxr-x  1 galahad galahad  216880 Mar 15 20:25 SG_RPROJ_TEST
-rwxrwxr-x  1 galahad galahad 1141608 Mar 15 20:25 SG_SGD_TEST
-rwxrwxr-x  1 galahad galahad  220176 Mar 15 20:25 SG_HOLTWINTERS_TEST
-rwxrwxr-x  1 galahad galahad  366344 Mar 15 20:25 SG_DBSCAN_TEST
-rwxrwxr-x  1 galahad galahad  436328 Mar 15 20:25 SG_LINKAGE_TEST
-rwxrwxr-x  1 galahad galahad  306712 Mar 15 20:25 SG_SHAP_KERNEL_TEST
-rwxrwxr-x  1 galahad galahad 1343976 Mar 15 20:25 SG_LARS_TEST
-rwxrwxr-x  1 galahad galahad  226880 Mar 15 20:25 SG_KMEANS_TEST
-rwxrwxr-x  1 galahad galahad  175712 Mar 15 20:25 SG_FIL_CHILD_INDEX_TEST
-rwxrwxr-x  1 galahad galahad  101192 Mar 15 20:25 SG_TRUSTWORTHINESS_TEST
-rwxrwxr-x  1 galahad galahad  419552 Mar 15 20:25 SG_GENETIC_NODE_TEST
-rwxrwxr-x  1 galahad galahad   92232 Mar 15 20:25 SG_GENETIC_PARAM_TEST
-rwxrwxr-x  1 galahad galahad   51248 Mar 15 20:25 SG_LOGGER_TEST
-rwxrwxr-x  1 galahad galahad  849008 Mar 15 20:25 SG_CD_TEST
-rwxrwxr-x  1 galahad galahad 1031408 Mar 15 20:25 SG_RIDGE_TEST
-rwxrwxr-x  1 galahad galahad  985072 Mar 15 20:25 SG_PCA_TEST
-rwxrwxr-x  1 galahad galahad 1010680 Mar 15 20:25 SG_OLS_TEST
-rwxrwxr-x  1 galahad galahad  376928 Mar 15 20:25 SG_MULTI_SUM_TEST
-rwxrwxr-x  1 galahad galahad   85000 Mar 15 20:25 SG_FNV_HASH_TEST
```

Then, individual tests can be run
```bash
$ ./SG_SVC_TEST
Running main() from ../googletest/src/gtest_main.cc
[==========] Running 36 tests from 14 test suites.
[----------] Global test environment set-up.
[----------] 2 tests from WorkingSetTest/0, where TypeParam = float
[ RUN      ] WorkingSetTest/0.Init
[       OK ] WorkingSetTest/0.Init (614 ms)
[ RUN      ] WorkingSetTest/0.Select
[       OK ] WorkingSetTest/0.Select (1 ms)
[----------] 2 tests from WorkingSetTest/0 (615 ms total)

[----------] 2 tests from WorkingSetTest/1, where TypeParam = double
[ RUN      ] WorkingSetTest/1.Init
[       OK ] WorkingSetTest/1.Init (0 ms)
[ RUN      ] WorkingSetTest/1.Select
[       OK ] WorkingSetTest/1.Select (1 ms)
[----------] 2 tests from WorkingSetTest/1 (1 ms total)
...
```

To run cuML c++ benchmarks, in the `build` directory:
```bash
$ ./bench/sg_benchmark  # Single GPU benchmarks
```
Refer to `--help` option to know more on its usage

To run ml-prims C++ benchmarks (optional):
```bash
$ ./bench/prims_benchmark  # ml-prims benchmarks
```

### Python Unit Tests

To run Python tests:

```bash
$ pytest -v
```

If only the single GPU algos want to be run, then:

```bash
$ pytest --ignore=cuml/test/dask
```

If you want a list of the available tests:
```bash
$ pytest cuml/test --collect-only
```
