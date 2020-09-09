# cuML Python Package

This folder contains the Python and Cython code of the algorithms and ML primitives of cuML, that are distributed in the Python cuML package.

Contents:

- [Build Configuration](#build-configuration)
- [RAFT Integration in cuml.raft](#raft-integration-in-cumlraft)
- [Running Unit Tests](#build-requirements)

### Build Configuration

The build system uses setup.py for configuration and building.

cuML's setup.py can be configured through environment variables and command line arguments.

The environment variables are:

| Environment variable | Possible values | Default behavior if not set | Behavior |
| --- | --- | --- | --- |
| CUDA_HOME | path/to/cuda_toolkit | Inferred by location of `nvcc` | Optional variable allowing to manually specify location of the CUDA toolkit. |
| CUML_BUILD_PATH | path/to/libcuml_build_folder | Looked for in path_to_cuml_repo/cpp/build | Optional variable allowing to manually specify location of libcuml++ build folder. |
| RAFT_PATH | path/to/raft |  Looked for in path_to_cuml_repo/cpp/build, if not found clone  | Optional variable allowing to manually specify location of the RAFT Repository. |

The command line arguments (i.e. passed alongside `setup.py` when invoking, for
example `setup.py --singlegpu`) are:


| Argument | Behavior |
| --- | --- |
| clean --all | Cleans all Python and Cython artifacts, including pycache folders, .cpp files resulting of cythonization and compiled extensions. |
| --singlegpu | Option to build cuML without multiGPU algorithms. Removes dependency on nccl, libcumlcomms and ucx-py. |


### RAFT Integration in cuml.raft

RAFT's Python and Cython is located in the [RAFT repository](https://github.com/rapidsai/raft/python). It was designed to be included in projects as opposed to be distributed by itself, so at build time, **setup.py creates a symlink from cuML, located in `/python/cuml/raft/` to the Python folder of RAFT**.

For developers that need to modify RAFT code, please refer to the [RAFT Developer Guide](https://github.com/rapidsai/raft/blob/branch-0.15/BUILD.md#developer-guide) for recommendations.

To configure RAFT at build time:

1. If the environment variable `RAFT_PATH` points to the RAFT repo, then that will be used.
2. If there is a libcuml build folder that has cloned RAFT already, setup.py will use that RAFT. Location of this can be configured with the environment variable CUML_BUILD_PATH.
3. If none of the above happened, then setup.py will clone RAFT and use it directly.

The RAFT Python code gets included in the cuML build and distributable artifacts as if it was always present in the folder structure of cuML.

### Build Requirements

cuML's convenience [development yaml files](https://github.com/rapidsai/cuml/tree/branch-0.15/conda/environments) includes all dependencies required to build cuML.

To build cuML's Python package, the following dependencies are required:

- cudatoolkit version corresponding to system CUDA toolkit
- setuptools
- cython >= 0.29, < 0.30
- numpy
- cmake >= 3.14
- cudf version matching the cuML version
- libcuml version matching the cuML version
- libcuml={{ version }}
- cupy>=7,<8.0.0a0
- joblib >=0.11

Packages required for multigpu algorithms*:
- libcumlprims version matching the cuML version
- ucx-py version matching the cuML version
- dask-cudf version matching the cuML version
- nccl>=2.5
- dask>=2.12.0
- distributed>=2.12.0

* this can be avoided with `--singlegpu` argument flag.


### Python Tests

Python tests are based on the pytest library. To run them, from the `path_to_cuml/python/` folder, simply type `pytest`.


