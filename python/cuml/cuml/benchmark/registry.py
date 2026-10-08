# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog estimator import specifications for each benchmark backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EstimatorSpec:
    """Identify an estimator's import location and distribution package."""

    module: str
    name: str
    package: str


def _spec(module: str, name: str, package: str) -> EstimatorSpec:
    """Create an estimator import specification."""
    return EstimatorSpec(module, name, package)


_SKLEARN_MODULES = {
    "KMeans": "sklearn.cluster",
    "DBSCAN": "sklearn.cluster",
    "SpectralClustering": "sklearn.cluster",
    "AgglomerativeClustering": "sklearn.cluster",
    "EmpiricalCovariance": "sklearn.covariance",
    "LedoitWolf": "sklearn.covariance",
    "PCA": "sklearn.decomposition",
    "IncrementalPCA": "sklearn.decomposition",
    "TruncatedSVD": "sklearn.decomposition",
    "GaussianRandomProjection": "sklearn.random_projection",
    "SparseRandomProjection": "sklearn.random_projection",
    "NearestNeighbors": "sklearn.neighbors",
    "KNeighborsClassifier": "sklearn.neighbors",
    "KNeighborsRegressor": "sklearn.neighbors",
    "KernelDensity": "sklearn.neighbors",
    "LinearRegression": "sklearn.linear_model",
    "LogisticRegression": "sklearn.linear_model",
    "ElasticNet": "sklearn.linear_model",
    "Ridge": "sklearn.linear_model",
    "Lasso": "sklearn.linear_model",
    "KernelRidge": "sklearn.kernel_ridge",
    "RandomForestClassifier": "sklearn.ensemble",
    "RandomForestRegressor": "sklearn.ensemble",
    "TSNE": "sklearn.manifold",
    "SpectralEmbedding": "sklearn.manifold",
    "SVC": "sklearn.svm",
    "SVR": "sklearn.svm",
    "LinearSVC": "sklearn.svm",
    "LinearSVR": "sklearn.svm",
    "MultinomialNB": "sklearn.naive_bayes",
    "BernoulliNB": "sklearn.naive_bayes",
    "ComplementNB": "sklearn.naive_bayes",
    "CategoricalNB": "sklearn.naive_bayes",
    "GaussianNB": "sklearn.naive_bayes",
    "StandardScaler": "sklearn.preprocessing",
    "MinMaxScaler": "sklearn.preprocessing",
    "MaxAbsScaler": "sklearn.preprocessing",
    "Normalizer": "sklearn.preprocessing",
    "RobustScaler": "sklearn.preprocessing",
    "PolynomialFeatures": "sklearn.preprocessing",
    "Binarizer": "sklearn.preprocessing",
    "KBinsDiscretizer": "sklearn.preprocessing",
    "PowerTransformer": "sklearn.preprocessing",
    "QuantileTransformer": "sklearn.preprocessing",
    "OneHotEncoder": "sklearn.preprocessing",
    "OrdinalEncoder": "sklearn.preprocessing",
    "LabelEncoder": "sklearn.preprocessing",
    "LabelBinarizer": "sklearn.preprocessing",
    "TargetEncoder": "sklearn.preprocessing",
}

CPU_REGISTRY = {
    name: _spec(module, name, "scikit-learn")
    for name, module in _SKLEARN_MODULES.items()
}
CPU_REGISTRY.update(
    {
        "UMAP": _spec("umap", "UMAP", "umap-learn"),
        "HDBSCAN": _spec("hdbscan", "HDBSCAN", "hdbscan"),
    }
)

_CUML_MODULES = {
    **{n: m.replace("sklearn.", "cuml.") for n, m in _SKLEARN_MODULES.items()},
    "UMAP": "cuml.manifold",
    "HDBSCAN": "cuml.cluster",
}
# cuML exposes these from slightly different modules/namespaces.
_CUML_MODULES.update(
    {
        "GaussianRandomProjection": "cuml.random_projection",
        "SparseRandomProjection": "cuml.random_projection",
        "KernelRidge": "cuml.kernel_ridge",
    }
)
SG_REGISTRY = {
    name: _spec(module, name, "cuml") for name, module in _CUML_MODULES.items()
}

# These names are the union of the public override module ``__all__`` values.
ACCEL_ESTIMATORS = frozenset(
    {
        "KMeans",
        "DBSCAN",
        "SpectralClustering",
        "EmpiricalCovariance",
        "LedoitWolf",
        "PCA",
        "IncrementalPCA",
        "TruncatedSVD",
        "RandomForestRegressor",
        "RandomForestClassifier",
        "KernelRidge",
        "LinearRegression",
        "LogisticRegression",
        "ElasticNet",
        "Ridge",
        "Lasso",
        "SpectralEmbedding",
        "TSNE",
        "NearestNeighbors",
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "KernelDensity",
        "StandardScaler",
        "MinMaxScaler",
        "MaxAbsScaler",
        "PolynomialFeatures",
        "OneHotEncoder",
        "TargetEncoder",
        "LabelEncoder",
        "LabelBinarizer",
        "SVC",
        "SVR",
        "LinearSVC",
        "LinearSVR",
        "UMAP",
        "HDBSCAN",
    }
)
ACCEL_REGISTRY = {name: CPU_REGISTRY[name] for name in ACCEL_ESTIMATORS}

_DASK_MODULES = {
    "KMeans": "cuml.dask.cluster",
    "DBSCAN": "cuml.dask.cluster",
    "PCA": "cuml.dask.decomposition",
    "TruncatedSVD": "cuml.dask.decomposition",
    "NearestNeighbors": "cuml.dask.neighbors",
    "KNeighborsClassifier": "cuml.dask.neighbors",
    "KNeighborsRegressor": "cuml.dask.neighbors",
    "LinearRegression": "cuml.dask.linear_model",
    "LogisticRegression": "cuml.dask.linear_model",
    "ElasticNet": "cuml.dask.linear_model",
    "Lasso": "cuml.dask.linear_model",
    "Ridge": "cuml.dask.linear_model",
    "RandomForestClassifier": "cuml.dask.ensemble",
    "RandomForestRegressor": "cuml.dask.ensemble",
    "UMAP": "cuml.dask.manifold",
    "MultinomialNB": "cuml.dask.naive_bayes",
    "OneHotEncoder": "cuml.dask.preprocessing",
    "OrdinalEncoder": "cuml.dask.preprocessing",
    "LabelEncoder": "cuml.dask.preprocessing",
    "LabelBinarizer": "cuml.dask.preprocessing",
}
DASK_REGISTRY = {
    name: _spec(module, name, "cuml") for name, module in _DASK_MODULES.items()
}
