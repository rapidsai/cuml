#!/usr/bin/env python
# coding: utf-8

# # RAPIDS cuML 
# ## Performance, Boundaries, and Correctness Benchmarks
# 
# **Description:** This notebook provides a simple and unified means of benchmarking single GPU cuML algorithms against their skLearn counterparts with the `cuml.benchmark` package in RAPIDS cuML. This enables quick and simple measurements of performance, validation of correctness, and investigation of upper bounds.
# 
# Each benchmark returns a Pandas `DataFrame` with the results. At the end of the notebook, these results are used to draw charts and output to a CSV file. 
# 
# Please refer to the [table of contents](#table_of_contents) for algorithms available to be benchmarked with this notebook.

# In[ ]:


import cuml
import pandas as pd

from cuml.benchmark.runners import SpeedupComparisonRunner
from cuml.benchmark.algorithms import algorithm_by_name

import warnings
warnings.filterwarnings('ignore', 'Expected column ')

print(cuml.__version__)


# In[ ]:


N_REPS = 3  # Number of times each test is repeated

DATA_NEIGHBORHOODS = "blobs"
DATA_CLASSIFICATION = "classification"
DATA_REGRESSION = "regression"

INPUT_TYPE = "numpy"

benchmark_results = []


# In[ ]:


SMALL_ROW_SIZES = [2**x for x in range(14, 17)]
LARGE_ROW_SIZES = [2**x for x in range(18, 24, 2)]

SKINNY_FEATURES = [32, 256]
WIDE_FEATURES = [1000, 10000]

VERBOSE=True
RUN_CPU=True


# In[ ]:


def enrich_result(algorithm, runner, result):
    result["algo"] = algorithm
    result["dataset_name"] = runner.dataset_name
    result["input_type"] = runner.input_type
    return result

def execute_benchmark(algorithm, runner, verbose=VERBOSE, run_cpu=RUN_CPU, **kwargs):
    results = runner.run(algorithm_by_name(algorithm), verbose=verbose, run_cpu=run_cpu, **kwargs)
    results = [enrich_result(algorithm, runner, result) for result in results]
    benchmark_results.extend(results)


# ## Table of Contents<a id="table_of_contents"/>
# 
# ### Benchmarks
# 1. [Neighbors](#neighbors)<br>
#     1.1 [Nearest Neighbors - Brute Force](#nn_bruteforce)<br>
#     1.2 [KNeighborsClassifier](#kneighborsclassifier)<br>
#     1.3 [KNeighborsRegressor](#kneighborsregressor)<br>
# 2. [Clustering](#clustering)<br>
#     2.1 [DBSCAN - Brute Force](#dbscan_bruteforce)<br>
#     2.2 [K-Means](#kmeans)<br>
# 3. [Manifold Learning](#manifold_learning)<br>
#     3.1 [UMAP - Unsupervised](#umap_unsupervised)<br>
#     3.2 [UMAP - Supervised](#umap_supervised)<br>
#     3.3 [T-SNE](#tsne)<br>
# 4. [Linear Models](#linear_models)<br>
#     4.1 [Linear Regression](#linear_regression)<br>
#     4.2 [Logistic Regression](#logistic_regression)<br>
#     4.3 [Ridge Regression](#ridge_regression)<br>
#     4.4 [Lasso Regression](#lasso_regression)<br>
#     4.5 [ElasticNet Regression](#elasticnet_regression)<br>
#     4.6 [Mini-batch SGD Classifier](#minibatch_sgd_classifier)<br>
# 5. [Decomposition](#decomposition)<br>
#     5.1 [PCA](#pca)<br>
#     5.2 [Truncated SVD](#truncated_svd)<br>
# 6. [Ensemble](#ensemble)<br>
#     6.1 [Random Forest Classifier](#random_forest_classifier)<br>
#     6.2 [Random Forest Regressor](#random_forest_regressor)<br>
#     6.3 [FIL](#fil)<br>
#     6.4 [Sparse FIL](#sparse_fil)<br>
# 7. [Random Projection](#random_projection)<br>
#     7.1 [Gaussian Random Projection](#gaussian_random_projection)<br>
#     7.2 [Sparse Random Projection](#sparse_random_projection)<br>
# 8. [SVM](#svm)<br>
#     8.1 [SVC - Linear Kernel](#svc_linear_kernel)<br>
#     8.2 [SVC - RBF Kernel](#svc_rbf_kernel)<br>
#     8.3 [SVR - Linear Kernel](#svr_linear_kernel)<br>
#     8.4 [SVR - RBF Kernel](#svr_rbf_kernel)<br>
#     
# ### Chart & Store Results
# 9. [Convert to Pandas DataFrame](#convert_to_pandas)<br>
# 10. [Chart Results](#chart_results)<br>
# 11. [Output to CSV](#output_csv)<br>

# ## Neighbors<a id="neighbors"/>
# 

# ### Nearest Neighbors - Brute Force<a id="nn_bruteforce"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS,
)

execute_benchmark("NearestNeighbors", runner)


# ### KNeighborsClassifier<a id="kneighborsclassifier"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("KNeighborsClassifier", runner)


# ### KNeighborsRegressor<a id="kneighborsregressor"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("KNeighborsRegressor", runner)


# ## Clustering<a id="clustering"/>

# ### DBSCAN - Brute Force<a id="dbscan_bruteforce"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("DBSCAN", runner)


# ### K-means Clustering<a id="kmeans"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type="numpy",
    n_reps=N_REPS
)

execute_benchmark("KMeans", runner)


# ## Manifold Learning<a id="manifold_learning"/>

# ### UMAP - Unsupervised<a id="umap_unsupervised"/>
# CPU benchmark requires UMAP-learn

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=WIDE_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("UMAP-Unsupervised", runner)


# ### UMAP - Supervised<a id="umap_supervised"/>
# CPU benchmark requires UMAP-learn

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=WIDE_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("UMAP-Supervised", runner)


# ### T-SNE<a id="tsne"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES, 
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

# Due to extreme high runtime, the CPU benchmark 
# is disabled. Use run_cpu=True to re-enable. 

execute_benchmark("TSNE", runner, run_cpu=True)


# ## Linear Models<a id="linear_models"/>

# ### Linear Regression<a id="linear_regression"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("LinearRegression", runner)


# ### Logistic Regression<a id="logistic_regression"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("LogisticRegression", runner)


# ### Ridge Regression<a id="ridge_regression"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("Ridge", runner)


# ### Lasso Regression<a id="lasso_regression"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("Lasso", runner)


# ### ElasticNet Regression<a id="elasticnet_regression"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("ElasticNet", runner)


# ### Mini-batch SGD Classifier<a id="minibatch_sgd_classifier"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("MBSGDClassifier", runner)


# ## Decomposition<a id="decomposition"/>

# ### PCA<a id="pca"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=WIDE_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("PCA", runner)


# ### Truncated SVD<a id="truncated_svd"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=WIDE_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("TSVD", runner)


# ## Ensemble<a id="ensemble"/>

# ### Random Forest Classifier<a id="random_forest_classifier"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("RandomForestClassifier", runner)


# ### Random Forest Regressor<a id="random_forest_regressor"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("RandomForestRegressor", runner)


# ### FIL<a id="fil"/>
# CPU benchmark requires XGBoost Library

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("FIL", runner)


# ## Sparse FIL<a id="sparse_fil"/>
# Requires TreeLite library

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("Sparse-FIL-SKL", runner)


# ## Random Projection<a id="random_projection"/>

# ### Gaussian Random Projection<a id="gaussian_random_projection"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES,
    bench_dims=WIDE_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("GaussianRandomProjection", runner)


# ### Sparse Random Projection<a id="sparse_random_projection"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES,
    bench_dims=WIDE_FEATURES,
    dataset_name=DATA_NEIGHBORHOODS,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("SparseRandomProjection", runner)


# ## SVM<a id="svm"/>

# ### SVC - Linear Kernel<a id="svc_linear_kernel"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

# Due to extreme high runtime, the CPU benchmark 
# is disabled. Use run_cpu=True to re-enable. 

execute_benchmark("SVC-Linear", runner, run_cpu=True)


# ### SVC - RBF Kernel<a id="svc_rbf_kernel"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_CLASSIFICATION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

# Due to extreme high runtime, the CPU benchmark 
# is disabled. Use run_cpu=True to re-enable. 

execute_benchmark("SVC-RBF", runner, run_cpu=True)


# ### SVR - Linear Kernel<a id="svr_linear_kernel"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

# Due to extreme high runtime, the CPU benchmark 
# is disabled. Use run_cpu=True to re-enable. 

execute_benchmark("SVR-Linear", runner, run_cpu=False)


# ### SVR - RBF Kernel<a id="svr_rbf_kernel"/>

# In[ ]:


runner = cuml.benchmark.runners.SpeedupComparisonRunner(
    bench_rows=SMALL_ROW_SIZES, 
    bench_dims=SKINNY_FEATURES,
    dataset_name=DATA_REGRESSION,
    input_type=INPUT_TYPE,
    n_reps=N_REPS
)

execute_benchmark("SVR-RBF", runner)


# ## Charting & Storing Results<a id="charting_and_storing_results"/>

# ### Convert Results to Pandas DataFrame<a id="convert_to_pandas"/>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.DataFrame(benchmark_results)


# ### Chart Results<a id="chart_results"/>

# In[ ]:


def chart_single_algo_speedup(df, algorithm):
    df = df.loc[df.algo == algorithm]
    df = df.pivot(index="n_samples", columns="n_features", values="speedup")
    axes = df.plot.bar(title="%s Speedup" % algorithm)


# In[ ]:


def chart_all_algo_speedup(df):
    df = df[["algo", "n_samples", "speedup"]].groupby(["algo", "n_samples"]).mean()
    df.plot.bar()


# In[ ]:


chart_single_algo_speedup(df, "LinearRegression")


# In[ ]:


chart_all_algo_speedup(df)


# ### Output Results to CSV<a id="output_csv"/>

# In[ ]:


df.to_csv("benchmark_results.csv")

