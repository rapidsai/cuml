# Introduction
This folder contains some of the common components and computational primitives
that form part of the machine learning algorithms in cuML, and can be used
individually as well in the form of a header only library.

# Setup
## Dependencies (pre-requisites)
1. cmake (>= 3.12.4)
2. CUDA  (>= 9.2)
3. doxygen (>= 1.8.11) (only required to build doxygen docs)
4. graphviz (>= 2.38.0) (only required to build doxygen docs)

## Getting the ML primitives:
```bash
$ git clone --recursive https://github.com/rapidsai/cuml
```
The primitives are contained in the `ml-prims` sub-folder.

# Building tests
```bash
$ cd cuml/ml-prims
$ mkdir build
$ cd build
## build to specific GPU arch with -DGPU_ARCHS=70, to reduce compile time!
$ cmake ..
$ make -j
```

# Running tests
```bash
# build using above instructions
$ cd build
$ ./test/mlcommon_test
```

# Build doxygen docs
This needs doxygen and graphviz to be installed.
```bash
# build using above instructions
$ cd build
$ make doc
```

# External
The external folder inside ml-prims contains submodules that this project
depends on. Appropriate location flags for these dependencies will be
automatically populated in the main `CMakeLists.txt`. Current external
submodules are:
1. [CUTLASS](https://github.com/NVIDIA/cutlass)
2. [CUB](https://github.com/NVlabs/cub)
3. [Google Test](https://github.com/google/googletest)

# Memory layout
Information about needed memory layout in current implementation:
1. Memory storage for matrix is dense, and in both column-major and row-major. Please see individual file/function documentation to see which format is needed for each case.
2. Matrix is densely packed without any LDA
