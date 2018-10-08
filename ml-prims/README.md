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

Other details TBD!
