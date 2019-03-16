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

For additional examples, browse our complete [API documentation](https://rapidsai.github.io/projects/cuml/en/latest/index.html), or check out our more detailed [walkthrough notebooks](https://github.com/rapidsai/notebooks/tree/master/cuml).

### Supported Algorithms:

- Truncated Singular Value Decomposition (tSVD) - Single GPU, Muti-GPU in Conda CUDA 10 package
- Linear Regression (Ordinary Least Squares) - Single GPU, Muti-GPU in Conda CUDA 10 package
- Principal Component Analysis (PCA) - Single GPU
- Density-based spatial clustering of applications with noise (DBSCAN) - Single GPU
- K-Means Clustering - Multi-GPU
- K-Nearest Neighbors - Multi-GPU (Requires [Faiss](https://github.com/facebookresearch/faiss) installation to use)
- Ridge Regression - Single GPU
- Kalman Filter - Single GPU
- UMAP
- Stochastic Gradient Descent for linear regression, logistic regression, and linear svm with L1, L2, and elastic-net penalties.

Algorithms in progress:

- More Kalman Filter versions
- Lasso
- Elastic-Net
- Logistic Regression

More ML algorithms in cuML and more ML primitives in ml-prims are being worked on. Goals for future versions include more algorithms and multi-gpu versions of the algorithms and primitives.

## Installation

Ensure `libomp` and `libopenblas` are installed, for example via apt:
```bash
sudo apt install libopenblas-base libomp-dev
```

#### Conda
cuML can be installed using the `rapidsai` conda channel:
```bash
conda install -c nvidia -c rapidsai -c conda-forge -c pytorch -c defaults cuml
```

#### Pip
cuML can also be installed using pip. Select the package based on your version of CUDA.


```bash
# cuda 9.2
pip install cuml-cuda92

# cuda 10.0
pip install cuml-cuda100
```

#### Build/Install from Source
See build [instructions](BUILD.md)

## Contributing

Please use GitHub issues and pull requests to report bugs and add or request functionality.

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
