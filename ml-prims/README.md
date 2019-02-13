# Introduction

This folder contains some of the common  components and
computational primitives that form part of the machine learning algorithms in cuML,
and can be used individually as well in the form of a header only library.

# Setup
## Dependencies (pre-requisites)
1. git
2. cmake (>= 3.12.4)
3. CUDA  (>= 9.2)

## Getting the ML primitives:
```bash
$ git clone --recursive https://github.com/rapidsai/cuml
```

The primitives are contained in the ml-prims folder.

## In case you prefer working inside docker
This now comes with open-mpi built inside the container itself.
```bash
$ git clone https://github.com/teju85/dockerfiles
$ cd dockerfiles/ubuntu1604
$ make ml-dev
$ cd ../..
$ ./dockerfiles/scripts/launch -runas user ml:dev /bin/bash
container$ cd /work/cuml/ml-prims
```

# Building and executing tests
```bash
$ cd cuml/ml-prims
$ mkdir build
$ cd build
$ cmake .. ## Use specific GPU architecture with -DGPU_ARCHS=70,  to significantly reduce compile time!
$ make -j
```

To run the tests:

```bash
# build using above instructions
$ cd build
$ ./test/mlcommon_test
```

## External

The external folders inside ml-prims contain submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main CMakeLists.txt file for these.

Current external submodules are:

1. [CUTLASS](https://github.com/NVIDIA/cutlass)
2. [CUB](https://github.com/NVlabs/cub)
3. [Google Test](https://github.com/google/googletest)

Information about needed memory layout in current implementation:

1. Memory storage for matrix is dense, and in both column-major and row-major. Please see individual file/function documentation to see which format is needed for each case..
2. Matrix is densely packed without any LDA
