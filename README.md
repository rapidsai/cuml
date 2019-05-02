# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuML - GPU Machine Learning Algorithms</div>

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cuml/blob/master/README.md) ensure you are on the `master` branch.

cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other [RAPIDS](https://rapids.ai/) projects.

cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming.

As an example, the following Python snippet loads input and computes DBSCAN clusters, all on GPU:
```python
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

print(dbscan_float.labels_)
```

Output:
```
0    0
1    1
2    2
dtype: int32
```

For additional examples, browse our complete [API documentation](https://docs.rapids.ai/api/cuml/stable/), or check out our more detailed [walkthrough notebooks](https://github.com/rapidsai/notebooks/tree/master/cuml).

### Supported Algorithms:

| Algorithm | Scale | Notes |
| --- | --- | --- |
| Linear Regression (OLS) | Single GPU | Multi-GPU available in conda cuda10 package and [dask-cuml](http://github.com/rapidsai/dask-cuml) |
| Stochastic Gradient Descent | Single-GPU | for linear regression, logistic regression, and linear svm with L1, L2, and elastic-net penalties |
| Coordinate Descent | Single-GPU | |
| Ridge Regression | Single-GPU |
| UMAP | Single-GPU |
| Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | Single GPU |
| K-Means Clustering | Single-GPU |
| K-Nearest Neighbors (KNN) | Multi-GPU with [dask-cuml](http://github.com/rapidsai/dask-cuml) <br> Uses [Faiss](https://github.com/facebookresearch/faiss) |
| Principal Component Analysis (PCA) | Single GPU |
| Truncated Singular Value Decomposition (tSVD) | Single GPU | Multi-GPU available in conda cuda10 package |
| Linear Kalman Filter | Single-GPU |

---

More ML algorithms in cuML and more ML primitives in ml-prims are being worked on, among them t-sne, decision trees, random forests and others. Goals for future versions include more multi-gpu versions of the algorithms and primitives.

## Installation

1. Install NVIDIA drivers with CUDA 9.2 or 10.0 
2. Ensure `libomp` and `libopenblas` are installed, for example via apt:
```bash
sudo apt install libopenblas-base libomp-dev
```

#### Conda
cuML can be installed using the `rapidsai` conda channel:

CUDA 9.2
```bash

conda install -c nvidia -c rapidsai -c conda-forge -c defaults cuml cudatoolkit=9.2
```

CUDA 10.0
```bash
conda install -c nvidia -c rapidsai -c conda-forge -c defaults cuml cudatoolkit=10.0
```

## Build/Install from Source
See the build [guide](BUILD.md).

## Contributing

Please see our [guide for contributing to cuML](CONTRIBUTING.md).

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
