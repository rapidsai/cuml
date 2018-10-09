# cuSKL

**PRIVATE REPO ONLY FOR TEST**
This version of cuSKL repository is only meant as an experimental version to test repository building and other git issues.

Machine learning algorithms for use with cuDF:

1. python: Python gdf machine learning package
2. cuML: C++/CUDA machine learning algorithms
3. ml-prims: Low level machine learning primitives used in CuML.

## Setup

To use CuSKL, it must be cloned and built in an environment that already where the dependencies, including PyGDF, libgdf and their own dependencies, are already installed.

To clone:

```
git clone --recurse-submodules git@github.com:rapidsai/cuskl_beta.git
```

To build the python package, in the repository root folder:

```
cd python
python setup.py install
```

### Dependencies

1. zlib
2. cmake (>= 3.8, version 3.11.4 is recommended and there are issues with version 3.12)
3. CUDA SDK (>= 8.0)
4. Cython (>= 0.28)
5. gcc (>=5.4.0)
6. nvcc
7. PyGDF
