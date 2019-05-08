# cuML
This repo contains some of the ML algorithms.

# Setup
## Dependencies

1. zlib
2. cmake (>= 3.8 and <= 3.11.4, version 3.11.4 is recommended and there are some issues with version 3.12)
3. CUDA SDK (>= 9.2)
4. Cython (>= 0.28)
5. gcc (>=5.4.0)
6. nvcc (this comes with CUDA SDK)

### Building cuML:

cuML is implemented as header only C++/CUDA libraries for the developers who would like to call these APIs from their projects. You can build and run the Google tests if you are interested in helping us to improve these libraries.

First, clone the cuML if you haven't cloned it yet.

```bash
$ git clone --recursive git@github.com:rapidsai/cuml-alpha.git
```

To build ml-prims, in the main folder;

```bash
$ cd cuML
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ ./ml_test
```

## External

The external folders inside cuML contain submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main CMakeLists.txt file for these.

Current external submodules are:

1. [CUTLASS](https://github.com/NVIDIA/cutlass)
2. [CUB](https://github.com/NVlabs/cub)
3. [Google Test](https://github.com/google/googletest)
