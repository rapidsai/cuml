# cuML (v0.1 Alpha)

Machine learning is a fundamental capability of RAPIDS. cuML is a suite of libraries that implements a machine learning algorithms within the RAPIDS data science ecosystem. cuML enables data scientists, researchers, and software engineers to run traditional ML tasks on  GPUs without going into the details of CUDA programming.

The cuML repository contains:

1. ***python***: Python based GPU Dataframe (GDF) machine learning package that takes [cuDF](https://github.com/rapidsai/cudf-alpha) dataframes as input. cuML connects the data to C++/CUDA based cuML and ml-prims libraries without ever leaving GPU memory.

2. ***cuML***: C++/CUDA machine learning algorithms. This library currently includes the following five algorithms;
   a. Single GPU Truncated Singular Value Decomposition (tSVD).
   b. Single GPU Principal Component Analysis (PCA).
   c. Single GPU Density-based Spatial Clustering of Applications with Noise (DBSCAN).
   d. Single GPU Kalman Filtering
   e. Multi-GPU K-Means Clustering

3. ***ml-prims***: Low level machine learning primitives used in cuML. ml-prims is comprised of the following components;
   a. Linear Algebra
   b. Statistics
   c. Basic Matrix Operations
   d. Distance Functions
   e. Random Number Generation

#### Available Algorithms for version 0.1alpha:

- Truncated Singular Value Decomposition (tSVD)

- Principal Component Analysis (PCA)

- Density-based spatial clustering of applications with noise (DBSCAN)

Upcoming algorithms for version 0.1:

- kmeans

- Kalman Filter

More ML algorithms in cuML and more ML primitives in ml-prims are being added currently. Example notebooks are provided in the python folder to test the functionality and performance of this v0.1 alpha version. Goals for future versions include more algorithms and  multi-gpu versions of the algorithms and primitives.

The installation option provided currently consists on building from source. Upcoming versions will add `pip` and `conda` options, along docker containers. They will be available in the coming weeks.


## Setup

### Dependencies

To use cuML, it must be cloned and built in an environment that already has the dependencies, including [cuDF](https://github.com/rapidsai/cudf-alpha) and its dependencies.

List of dependencies:

1. zlib
2. cmake (>= 3.8, version 3.11.4 is recommended and there are issues with version 3.12)
3. CUDA SDK (>= 8.0)
4. Cython (>= 0.28)
5. gcc (>=5.4.0)
6. nvcc
7. [cuDF](https://github.com/gpuopenanalytics/pygdf)

### Setup steps

To clone:

```
git clone --recurse-submodules git@github.com:rapidsai/cuskl-alpha.git
```

To build the python package, in the repository root folder:

```
cd python
python setup.py install
```

### Building CuML:

### Running tests

To test the C++ algorithms using googletests, without compiling the

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ ./ml_test
```

### Python Tests

Additional python tests can be found in the pythontests folder, along some useful scripts. Py.test based unit testing are being added for the final version 0.1.

## External

The external folders contains submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main CMakeLists.txt file for these.

Current external subbmodules are:

- (CUTLASS)[https://github.com/NVIDIA/cutlass]
- (Google Test)[https://github.com/google/googletest]
- (CUB)[https://github.com/NVlabs/cub]


## Contributing

Section in progress


## Contact:


