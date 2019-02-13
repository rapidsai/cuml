# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuML - Machine Learning Algorithms</div>

cuML is a suite of libraries that implement machine learning algorithms and share compatible APIs with other [RAPIDS](https://rapids.ai/) projects.

cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming.

As an example, the following Python snippet loads input and computes DBSCAN clusters, all on GPU:
```
import cudf
from cuml import DBSCAN

# Create and populate a GPU DataFrame
gdf_float = cudf.DataFrame()
gdf_float['0'] = [1.0, 2.0, 5.0]
gdf_float['1'] = [4.0, 2.0, 1.0]
gdf_float['2'] = [4.0, 2.0, 1.0]

# Setup and fit clusters
dbscan_float = DBSCAN(eps=1.0, min_samples=1)
dbscan_float.fit(gdf_float)

print(dbsca_float.labels_)
```

Output:
```
0    0
1    1
2    2
dtype: int32
```

For additional examples, browse our complete [API documentation](https://rapidsai.github.io/projects/cuml/en/latest/index.html), or check out our more detailed [walkthrough notebooks](https://github.com/rapidsai/notebooks/tree/master/cuml).

### Supported Algorithms:

- Truncated Singular Value Decomposition (tSVD)

- Principal Component Analysis (PCA)

- Density-based spatial clustering of applications with noise (DBSCAN)

- K-Means Clustering

- K-Nearest Neighbors (Requires [Faiss](https://github.com/facebookresearch/faiss) installation to use)

- Linear Regression (Ordinary Least Squares)

- Ridge Regression

- Kalman Filter


## Installation

#### Conda
cuML can be installed using the `rapidsai` conda channel:
```bash
conda install -c nvidia -c rapidsai -c conda-forge -c pytorch -c defaults cuml
```

#### Pip
cuML can also be installed using pip. Select the package based on your version of CUDA:
```bash
# cuda 9.2
pip install cuml-cuda92

# cuda 10.0
pip install cuml-cuda100
```
You also need to ensure `libomp` and `libopenblas` are installed:
```bash
apt install libopenblas-base libomp-dev
```

*Note:* Pip has no faiss-gpu package: If installing cuML from pip and you plan to use cuml.KNN, you must install [Faiss](https://github.com/facebookresearch/faiss) manually or via conda (see below).


**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cuml/blob/master/README.md) ensure you are on the `master` branch.

The cuML repository contains:

1. ***cuML***: C++/CUDA machine learning algorithms. This library currently includes the following six algorithms:
  a) Single GPU Truncated Singular Value Decomposition (tSVD)
  b) Single GPU Principal Component Analysis (PCA)
  c) Single GPU Density-based Spatial Clustering of Applications with Noise (DBSCAN)
  d) Single GPU Kalman Filtering
  e) Multi-GPU K-Means Clustering
  f) Multi-GPU K-Nearest Neighbors (Uses [Faiss](https://github.com/facebookresearch/faiss))

2. ***python***: Python bindings for the above, including interfaces for [cuDF](https://github.com/rapidsai/cudf) GPU dataframes. cuML connects the data to C++/CUDA based cuML and ml-prims libraries without ever leaving GPU memory.

3. ***ml-prims***: Low level machine learning primitives used in cuML. ml-prims is comprised of the following components:
  a) Linear Algebra
  b) Statistics
  c) Basic Matrix Operations
  d) Distance Functions
  e) Random Number Generation

Algorithms in progress:

- More Kalman Filter versions
- Lasso
- Elastic-Net
- Logistic Regression
- UMAP


More ML algorithms in cuML and more ML primitives in ml-prims are being worked on. Goals for future versions include more algorithms and multi-gpu versions of the algorithms and primitives.

### Build from Source
See [docs/build.md]

## External

The external folders contains submodules that this project in-turn depends on. Appropriate location flags
will be automatically populated in the main `CMakeLists.txt` file for these.

Current external submodules are:

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Google Test](https://github.com/google/googletest)
- [CUB](https://github.com/NVlabs/cub)

## Contributing

Please use GitHub issues and pull requests to report bugs and add or request functionality.

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)


## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
