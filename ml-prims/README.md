# ml-prims
This repo contains most of the ML primitives.

# Setup
## Dependencies

1. zlib
2. cmake (>= 3.8 and <= 3.11.4, version 3.11.4 is recommended and there are some issues with version 3.12)
3. CUDA SDK (>= 8.0)
4. Cython (>= 0.28)
5. gcc (>=5.4.0)
6. nvcc (this comes with CUDA SDK)

### Building ml-prims:

ml-prims is implemented as header only C++/CUDA libraries for the developers who would like to call these APIs from their projects. You can build and run the Google tests if you are interested in helping us to improve these libraries.

First, clone the cuSKL if you haven't cloned it yet.

```bash
$ git clone --recursive https://gitlab-master.nvidia.com/RAPIDS/cuSKL
```

To build ml-prims, in the main folder;

```bash
$ cd ml-prims
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ ./mlcommon_test
```

## External

The external folders inside ml-prims contain submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main CMakeLists.txt file for these.

Current external submodules are:

1. [CUTLASS](https://github.com/NVIDIA/cutlass)
2. [CUB](https://github.com/NVlabs/cub)
3. [Google Test](https://github.com/google/googletest)
