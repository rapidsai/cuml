# cuML Build From Source Guide

## Setting Up Your Build Environment

To install cuML from source, ensure the following dependencies are met:

**Hardware Needed to Run cuML:**
To run cuML code, you will need an NVIDIA GPU with compute capability 7.0 or higher (Volta™ architecture or newer). Note that while a GPU is not required to build or develop cuML itself, it is necessary to execute and test GPU-accelerated functionality.

**Software Dependencies:**
1. CUDA Toolkit (>= 12.0) - must include development libraries (cudart, cublas, cusparse, cusolver, curand, cufft)
2. gcc (>= 13.0)
3. cmake (>= 3.30.4)
4. ninja - build system used by default
5. Python (>= 3.10 and <= 3.13)
6. Cython (>= 3.0.0)

**RAPIDS Ecosystem Libraries:**

These RAPIDS libraries must match the cuML version (e.g., all version 25.10 if building cuML 25.10):

*C++ Libraries:*
- [librmm](https://github.com/rapidsai/rmm) - RAPIDS Memory Manager (C++ library)
- [libraft](https://github.com/rapidsai/raft) - RAPIDS CUDA accelerated algorithms (C++ library)
- [libraft-headers](https://github.com/rapidsai/raft) - RAPIDS RAFT headers
- [libcuvs](https://github.com/rapidsai/cuvs) - CUDA Vector Search library
- [libcumlprims](https://github.com/rapidsai/cuml) - cuML primitives

*Python Packages:*
- [rmm](https://github.com/rapidsai/rmm) - RAPIDS Memory Manager (Python package)
- [pylibraft](https://github.com/rapidsai/raft) - RAPIDS CUDA accelerated algorithms (Python package)
- [cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame library (Python package)

**Python Runtime Dependencies:**
- scikit-learn (>= 1.4)
- scipy (>= 1.8.0)
- joblib (>= 0.11)
- numpy (>= 1.23, < 3.0)
- cuda-python (>= 12.9.2, < 13.0a0 for CUDA 12.x; >= 13.0.1, < 14.0a0 for CUDA 13.x)

**Python Build Dependencies:**
- scikit-build-core (>= 0.10.0)
- rapids-build-backend (>= 0.4.0, < 0.5.0.dev0)

**Other External Libraries:**
- treelite (4.4.1 or compatible) - tree model library for model deployment
- rapids-logger (0.2.x) - logging library

**Multi-GPU Support:**

cuML has limited support for multi-GPU and multi-node operations. The following dependencies enable these features:

- **NCCL** (>= 2.19) - required for multi-GPU communication; provided automatically as a transitive dependency through libcuvs (listed above)
- **UCX** (>= 1.7) - optional; only required for multi-node operations (not needed for multi-GPU on a single node); must be explicitly enabled during build with `WITH_UCX=ON` (see [Using Infiniband for MNMG](wiki/mnmg/Using_Infiniband_for_MNMG.md))

**For development only:**
- clang-format (= 20.1.4) - enforces uniform C++ coding style; required for pre-commit hooks and CI checks. The packages `clang=20` and `clang-tools=20` from the conda-forge channel should be sufficient, if you are using conda. If not using conda, install the right version using your OS package manager.

It is recommended to use conda for environment/package management. If doing so, development environment .yaml files are located in `conda/environments/all_*.yaml`. These files contain most of the dependencies mentioned above. To create a development environment named `cuml_dev`, you can use the following commands (adjust the YAML filename to match your CUDA version and architecture):

```bash
conda create -n cuml_dev python=3.13
conda env update -n cuml_dev --file=conda/environments/all_cuda-130_arch-x86_64.yaml
conda activate cuml_dev
```

## Installing from Source

### Recommended Process

As a convenience, a `build.sh` script is provided to simplify the build process. The libraries will be installed to `$INSTALL_PREFIX` if set (e.g., `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ ./build.sh                           # build the cuML libraries, tests, and python package, then
                                       # install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```
For workflows that involve frequent switching among branches or between debug and release builds, it is recommended that you install [ccache](https://ccache.dev/) and make use of it by passing the `--ccache` flag to `build.sh`.

To build individual components, specify them as arguments to `build.sh`:
```bash
$ ./build.sh libcuml                   # build and install the cuML C++ and C-wrapper libraries
$ ./build.sh cuml                      # build and install the cuML Python package
$ ./build.sh prims                     # build the ml-prims tests
$ ./build.sh bench                     # build the cuML C++ benchmark
$ ./build.sh prims-bench               # build the ml-prims C++ benchmark
```

Other `build.sh` options:
```bash
$ ./build.sh clean                     # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcuml -v                # build and install libcuml with verbose output
$ ./build.sh libcuml -g                # build and install libcuml for debug
$ PARALLEL_LEVEL=8 ./build.sh libcuml  # build and install libcuml limiting parallel build jobs to 8 (ninja -j8)
$ ./build.sh libcuml -n                # build libcuml but do not install
$ ./build.sh prims --allgpuarch        # build the ML prims tests for all supported GPU architectures
$ ./build.sh cuml --singlegpu          # build the cuML Python package without MNMG algorithms
$ ./build.sh --ccache                  # use ccache to cache compilations, speeding up subsequent builds
```

By default, Ninja is used as the cmake generator. To override this and use, e.g., `make`, define the `CMAKE_GENERATOR` environment variable accordingly:
```bash
CMAKE_GENERATOR='Unix Makefiles' ./build.sh
```

To run the C++ unit tests (optional), from the repo root:

```bash
$ cd cpp/build
$ ./test/ml # single-GPU algorithm tests
$ ./test/ml_mg # multi-GPU algorithm tests, if --singlegpu was not used
$ ./test/prims # ML Primitive function tests
```

If you want a list of the available C++ tests:
```bash
$ ./test/ml --gtest_list_tests # single-GPU algorithm tests
$ ./test/ml_mg --gtest_list_tests # multi-GPU algorithm tests
$ ./test/prims --gtest_list_tests # ML Primitive function tests
```

To run all Python tests, including multi-GPU algorithms, from the repo root:
```bash
$ cd python
$ pytest -v
```

To run only single-GPU algorithm tests:

```bash
$ pytest --ignore=cuml/tests/dask --ignore=cuml/tests/test_nccl.py
```

If you want a list of the available Python tests:
```bash
$ pytest cuml/tests --collect-only
```

**Note:** Some tests require `xgboost`. If running tests in conda devcontainers, you must install the `xgboost` conda package manually. See `dependencies.yaml` for version information.

### Manual Process

Once dependencies are present, follow the steps below:

1. Clone the repository:
```bash
$ git clone https://github.com/rapidsai/cuml.git
```

2. Build and install `libcuml++` (C++/CUDA library containing the cuML algorithms), starting from the repository root folder:
```bash
$ cd cpp
$ mkdir build && cd build
$ cmake ..
```

**Note:** If CUDA is not in your PATH, you may need to set `CUDA_BIN_PATH` before running cmake:
```bash
$ export CUDA_BIN_PATH=$CUDA_HOME  # Default: /usr/local/cuda
```

If using a conda environment (recommended), configure cmake for `libcuml++`:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```

**Note:** You may see the following warning depending on your cmake version and `CMAKE_INSTALL_PREFIX`. This warning can be safely ignored:
```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```
To silence it, add `-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib` to your `cmake` command.

To reduce compile times, you can specify GPU compute capabilities to compile for. For example, for Volta GPUs:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="70"
```

Or for multiple architectures (e.g., Ampere and Hopper):

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
```

You may also wish to make use of `ccache` to reduce build times when switching among branches or between debug and release builds:

```bash
$ cmake .. -DUSE_CCACHE=ON
```

There are many options to configure the build process, see the [customizing build section](#custom-build-options).

3. Build `libcuml++` and `libcuml`:

```bash
$ make -j
$ make install
```

To run tests (optional):
```bash
$ ./test/ml # single-GPU algorithm tests
$ ./test/ml_mg # multi-GPU algorithm tests
$ ./test/prims # ML Primitive function tests
```

If you want a list of the available tests:
```bash
$ ./test/ml --gtest_list_tests # single-GPU algorithm tests
$ ./test/ml_mg --gtest_list_tests # multi-GPU algorithm tests
$ ./test/prims --gtest_list_tests # ML Primitive function tests
```

To run cuML C++ benchmarks (optional):
```bash
$ ./bench/sg_benchmark  # single-GPU benchmarks
```
Use the `--help` option for more information.

To run ml-prims C++ benchmarks (optional):
```bash
$ ./bench/prims_benchmark  # ml-prims benchmarks
```
Use the `--help` option for more information.

To build doxygen docs for all C/C++ source files:
```bash
$ make doc
```

4. Build and install the `cuml` python package:

From the repository root:
```bash
$ python -m pip install --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true python/cuml
```

To run Python tests (optional):

```bash
$ cd python
$ pytest -v
```

To run only single-GPU algorithm tests:

```bash
$ pytest --ignore=cuml/tests/dask --ignore=cuml/tests/test_nccl.py
```

If you want a list of the available tests:
```bash
$ pytest cuml/tests --collect-only
```

### Custom Build Options

#### libcuml & libcuml++

cuML's cmake has the following configurable flags available:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BUILD_CUML_CPP_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml++ shared library. Setting this variable to `OFF` sets the variables BUILD_CUML_C_LIBRARY, BUILD_CUML_TESTS, BUILD_CUML_MG_TESTS and BUILD_CUML_EXAMPLES to `OFF` |
| BUILD_CUML_C_LIBRARY | [ON, OFF]  | ON  | Enable/disable building libcuml shared library. Setting this variable to `ON` will set the variable BUILD_CUML_CPP_LIBRARY to `ON` |
| BUILD_CUML_STD_COMMS | [ON, OFF] | ON | Enable/disable building cuML NCCL+UCX communicator for running multi-node multi-GPU algorithms. Note that UCX support can also be enabled/disabled (see below). Note that BUILD_CUML_STD_COMMS and BUILD_CUML_MPI_COMMS are not mutually exclusive and can both be installed simultaneously. |
| WITH_UCX | [ON, OFF] | OFF | Enable/disable UCX support for the standard cuML communicator. Algorithms requiring point-to-point messaging will not work when this is disabled. This has no effect on the MPI communicator. |
| BUILD_CUML_MPI_COMMS | [ON, OFF] | OFF | Enable/disable building cuML MPI+NCCL communicator for running multi-node multi-GPU C++ tests. Note that BUILD_CUML_STD_COMMS and BUILD_CUML_MPI_COMMS are not mutually exclusive, and can both be installed simultaneously. |
| BUILD_CUML_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_test`.  |
| BUILD_CUML_MG_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuML algorithm test executable `ml_mg_test`. |
| BUILD_PRIMS_TESTS | [ON, OFF]  | ON  | Enable/disable building cuML algorithm test executable `prims_test`.  |
| BUILD_CUML_EXAMPLES | [ON, OFF]  | ON  | Enable/disable building cuML C++ API usage examples.  |
| BUILD_CUML_BENCH | [ON, OFF] | ON | Enable/disable building of cuML C++ benchmark.  |
| CMAKE_CXX11_ABI | [ON, OFF]  | ON  | Enable/disable the GLIBCXX11 ABI  |
| DETECT_CONDA_ENV | [ON, OFF] | ON | Use detection of conda environment for dependencies. If set to ON, and no value for CMAKE_INSTALL_PREFIX is passed, then it will assign it to $CONDA_PREFIX (to install in the active environment).  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| CMAKE_CUDA_ARCHITECTURES |  List of GPU architectures, semicolon-separated | Empty  | List the GPU architectures to compile the GPU targets for. Set to "NATIVE" to auto detect GPU architecture of the system, set to "ALL" to compile for all RAPIDS supported archs.  |
| KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuml++.  |
