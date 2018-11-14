# Introduction
This repo contains some of the common infrastructural components as well as
computational primitives, that will be useful while building a ML algo repo from
scratch.

# Setup
## Dependencies
1. git
2. zlib
3. cmake (>= 3.8)
4. CUDA SDK (>= 8.0)

## Repo
```bash
$ git clone --recursive git@gitlab.com:nvdevtech/ml-common.git
$ git submodule init
$ git submodule update
```

## In case you prefer working inside docker
```bash
$ git clone https://github.com/teju85/dockerfiles
$ cd dockerfiles/ubuntu1604
$ make ml-dev
$ cd ../..
$ ./dockerfiles/scripts/launch -runas user ml:dev /bin/bash
container$ cd /work/ml-common
```

# Running tests
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ ./mlcommon_test
```

# Users
## scripts
Contains some useful scripts. Refer to [scripts](scripts/README.md).

## external
Contains submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main CMakeLists.txt file, for these.

# Developers
## Contributing
Refer to Keith's excellent document on
[RAPIDS Git Workflow](https://docs.google.com/document/d/1oWUT8tdADaVxSVuvwtUfWtI0rLeKW80SX-Vnxgqq6ZQ/edit).

## Adding benchmarking tests
### Introduction
The goal here is to define and run directed tests aimed at measuring the performance
of our kernels in this repo and also track their SOL%. This will in turn help us in
improving perf over time. Whereas the unit-tests are aimed at verifying functional
correctness of our kernels, these benchmark tests try to closely mirror the
dimensions of real use-cases and measure perf of these kernels (stand-alone!).

### Running benchmark tests
Pretty much the same as above.
```bash
$ mkdir build
$ cd build
$ cmake ..  ## Use -DGPU_ARCHS=70, for eg, to reduce compile time!
$ make -j
$ ./mlcommon_bench
```

### Adding a new benchmark test
TLDR version:
1. Define a 'Params' struct which will contain all info about the workload 'sizes'
2. Define a subclass of 'Benchmark' class and implement its methods accordingly
3. At the end call the macro 'REGISTER_BENCH' to setup this test for running
For more detailed information, refer to the files harness.[h|cu]. There's also
an existing benchmark written for the 'add' kernel. Copying from it could also
be a good starting point!
