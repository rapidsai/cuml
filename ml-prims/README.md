# Introduction
This repo contains some of the common infrastructural components as well as
computational primitives, that will be useful while building a ML algo repo from
scratch.

# Setup
## Dependencies (pre-requisites)
1. git
2. zlib
3. cmake (>= 3.8)
4. CUDA SDK (>= 8.0)

## Repo
```bash
$ git clone --recursive https://github.com/rapidsai/cuml
```

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

# Building all test executables
```bash
$ cd cuml/ml-prims
$ mkdir build
$ cd build
$ cmake .. ## Use -DGPU_ARCHS=70, for eg, to reduce compile time!
$ make -j
```

# Running tests on single-gpu prims
```bash
# build using above instructions
$ cd build
$ ./test/mlcommon_test
```
