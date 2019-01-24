# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuML - RAPIDS Machine Learning Algorithms</div>

Machine learning is a fundamental capability of RAPIDS. cuML is a suite of libraries that implements a machine learning algorithms within the RAPIDS data science ecosystem. cuML enables data scientists, researchers, and software engineers to run traditional ML tasks on GPUs without going into the details of CUDA programming.

The cuML repository contains:

1. ***python***: Python based GPU Dataframe (GDF) machine learning package that takes [cuDF](https://github.com/rapidsai/cudf) dataframes as input. cuML connects the data to C++/CUDA based cuML and ml-prims libraries without ever leaving GPU memory.

2. ***cuML***: C++/CUDA machine learning algorithms. This library currently includes the following six algorithms;
   a) Single GPU Truncated Singular Value Decomposition (tSVD),
   b) Single GPU Principal Component Analysis (PCA),
   c) Single GPU Density-based Spatial Clustering of Applications with Noise (DBSCAN),
   d) Single GPU Kalman Filtering,
   e) Multi-GPU K-Means Clustering,
   f) Multi-GPU K-Nearest Neighbors (Uses [Faiss](https://github.com/facebookresearch/faiss)).

3. ***ml-prims***: Low level machine learning primitives used in cuML. ml-prims is comprised of the following components;
   a) Linear Algebra,
   b) Statistics,
   c) Basic Matrix Operations,
   d) Distance Functions,
   e) Random Number Generation.

#### Available Algorithms:

- Truncated Singular Value Decomposition (tSVD),

- Principal Component Analysis (PCA),

- Density-based spatial clustering of applications with noise (DBSCAN),

- K-Means Clustering,

- K-Nearest Neighbors (Requires [Faiss](https://github.com/facebookresearch/faiss) installation to use),

- Linear Regression (Ordinary Least Squares),

- Ridge Regression.

- Kalman Filter.

Upcoming algorithms:

- More Kalman Filter versions, 

- Lasso,

- Elastic-Net,

- Logistic Regression,

- UMAP


More ML algorithms in cuML and more ML primitives in ml-prims are being added currently. Example notebooks are provided in the python folder to test the functionality and performance. Goals for future versions include more algorithms and multi-gpu versions of the algorithms and primitives.

The installation option provided currently consists on building from source. Upcoming versions will add `pip` and `conda` options, along docker containers. They will be available in the coming weeks.


## Setup

### Conda
cuML can be installed using the `rapidsai` conda channel:
```
conda install -c nvidia -c rapidsai -c conda-forge -c pytorch -c defaults cuml
```

### Pip
cuML can also be installed using pip. Select the package based on your version of CUDA:
```
pip install cuml-cuda92
pip install cuml-cuda100
```
You also need to ensure `libomp` and `libopenblas` are installed:
```
apt install libopenblas-base libomp-dev
```

*Note:* There is no faiss-gpu package installable by pip, so the KNN algorithm will not work unless you install [Faiss](https://github.com/facebookresearch/faiss) manually or via conda (see below).

### Dependencies for Installing/Building from Source:

To install cuML from source, ensure the dependencies are met:


1. [cuDF](https://github.com/rapidsai/cudf) (>=0.4.0)
2. zlib Provided by zlib1g-dev in Ubuntu 16.04
3. cmake (>= 3.8, version 3.11.4 or 3.12.4 is recommended)
4. CUDA (>= 9.2)
5. Cython (>= 0.28)
6. gcc (>=5.4.0)
7. faiss-gpu (>=1.4.0) - faiss-gpu is required to run the KNN algorithm. For using KNN with CUDA 9.2, faiss-gpu can be installed using conda:

```conda install -c pytorch faiss-gpu cuda92```

To use KNN on CUDA 10, faiss-gpu needs to be installed from source, with other installation options coming soon.

### Installing from Source:

Once dependencies are present, follow the steps below:

1. Clone the repository.
```bash
$ git clone --recurse-submodules https://github.com/rapidsai/cuml.git
```

2. Build and install `libcuml` (the C++/CUDA library containing the cuML algorithms), starting from the repository root folder:
```bash
$ cd cuML
$ mkdir build
$ cd build
$ cmake ..
```

Note: if using a conda environment (recommended currently), then cmake can be configured appropriately via:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```

3. Build `libcuml`:

```bash
$ make -j
$ make install
```

To run tests (optional):

```bash
$ ./ml_test
```

If you want a list of the available tests:
```bash
$ ./ml_test --gtest_list_tests
```

4. Build the `cuml` python package:

```bash
$ cd ../../python
$ python setup.py build_ext --inplace
```

To run Python tests (optional):

```bash
$ py.test -v
```

If you want a list of the available tests:
```bash
$ py.test cuML/test --collect-only
```

5. Finally, install the Python package to your Python path:

```bash
$ python setup.py install
```

### Python Notebooks

Demo notebooks can be found in `python/notebooks` folder.

## External

The external folders contains submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main `CMakeLists.txt` file for these.

Current external submodules are:

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Google Test](https://github.com/google/googletest)
- [CUB](https://github.com/NVlabs/cub)

## Contributing

Please use issues and pull requests to report bugs and add functionality.

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)


## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
