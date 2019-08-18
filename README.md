# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuML - GPU Machine Learning Algorithms</div>

cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other [RAPIDS](https://rapids.ai/) projects.

cuML enables data scientists, researchers, and software engineers to run
traditional tabular ML tasks on GPUs without going into the details of CUDA
programming. In most cases, cuML's Python API matches the API from
[scikit-learn](https://scikit-learn.org).


For large datasets, these GPU-based implementations can complete 10-50x faster
than their CPU equivalents. For details on performance, see the [cuML Benchmarks
Notebook](https://github.com/rapidsai/notebooks-contrib/blob/master/intermediate_notebooks/benchmarks/cuml_benchmarks.ipynb).

As an example, the following Python snippet loads input and computes DBSCAN clusters, all on GPU:
```python
import cudf
from cuml.cluster import DBSCAN

# Create and populate a GPU DataFrame
gdf_float = cudf.DataFrame()
gdf_float['0'] = [1.0, 2.0, 5.0]
gdf_float['1'] = [4.0, 2.0, 1.0]
gdf_float['2'] = [4.0, 2.0, 1.0]

# Setup and fit clusters
dbscan_float = DBSCAN(eps=1.0, min_samples=1)
dbscan_float.fit(gdf_float)

print(dbscan_float.labels_)
```

Output:
```
0    0
1    1
2    2
dtype: int32
```

cuML also features multi-GPU and multi-node-multi-GPU operation, using [Dask](https://www.dask.org), for a 
growing list of algorithms. The following Python snippet reads input from a CSV file and performs 
a NearestNeighbors query across a cluster of Dask workers, using multiple GPUs on a single node:
```python
# Create a Dask CUDA cluster w/ one worker per device
from dask_cuda import LocalCUDACluster
cluster = LocalCUDACluster()

# Read CSV file in parallel across workers
import dask_cudf
df = dask_cudf.read_csv("/path/to/csv")

# Fit a NearestNeighbors model and query it
from cuml.dask.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors = 10)
nn.fit(df)
neighbors = nn.kneighbors(df)
```


For additional examples, browse our complete [API
documentation](https://docs.rapids.ai/api/cuml/stable/), or check out our
introductory [walkthrough
notebooks](https://github.com/rapidsai/notebooks/tree/master/cuml). Finally, you
can find complete end-to-end examples in the [notebooks-contrib
repo](https://github.com/rapidsai/notebooks-contrib).


### Supported Algorithms
| Category | Algorithm | Notes |
| --- | --- | --- |
| **Clustering** |  Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | |
|  | K-Means | Multi-Node Multi-GPU |
| **Dimensionality Reduction** | Principal Components Analysis (PCA) | |
| | Truncated Singular Value Decomposition (tSVD) | Multi-GPU version available (CUDA 10 only) |
| | Uniform Manifold Approximation and Projection (UMAP) | |
| | Random Projection | |
| | t-Distributed Stochastic Neighbor Embedding (TSNE) | (Experimental) |
| **Linear Models for Regression or Classification** | Linear Regression (OLS) | Multi-GPU available in conda CUDA 10 package |
| | Linear Regression with Lasso or Ridge Regularization | |
| | ElasticNet Regression | |
| | Logistic Regression | |
| | Stochastic Gradient Descent (SGD), Coordinate Descent (CD), and Quasi-Newton (QN) (including L-BFGS and OWL-QN) solvers for linear models  | |
| **Nonlinear Models for Regression or Classification** | Random Forest (RF) Classification | Experimental multi-node, multi-GPU version available via Dask integration |
| | Random Forest (RF) Regression | Experimental multi-node, multi-GPU version available via Dask integration |
|  | K-Nearest Neighbors (KNN) | Multi-GPU <br> Uses [Faiss](https://github.com/facebookresearch/faiss) |
| **Time Series** | Linear Kalman Filter | |
|  | Holt-Winters Exponential Smoothing | |
---

More ML algorithms in cuML and more ML primitives in ml-prims are planned for
future releases, including: spectral embedding, spectral clustering,
support vector machines, and additional time series methods. Future releases
will also expand support for multi-node, multi-GPU algorithms.

## Installation

See [the RAPIDS Release
Selector](https://rapids.ai/start.html#rapids-release-selector) for the command
line to install either nightly or official release cuML packages via Conda or
Docker.

## Build/Install from Source
See the build [guide](BUILD.md).

## Contributing

Please see our [guide for contributing to cuML](CONTRIBUTING.md).

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
