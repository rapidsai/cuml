#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# If libcuml was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcuml
except ModuleNotFoundError:
    pass
else:
    libcuml.load_library()
    del libcuml

from pylibraft.common import Handle

import cuml.feature_extraction
from cuml._version import __git_commit__, __version__
from cuml.cluster.agglomerative import AgglomerativeClustering
from cuml.cluster.dbscan import DBSCAN
from cuml.cluster.hdbscan import HDBSCAN
from cuml.cluster.kmeans import KMeans
from cuml.common.pointer_utils import device_of_gpu_matrix
from cuml.datasets.arima import make_arima
from cuml.datasets.blobs import make_blobs
from cuml.datasets.classification import make_classification
from cuml.datasets.regression import make_regression
from cuml.decomposition.incremental_pca import IncrementalPCA
from cuml.decomposition.pca import PCA
from cuml.decomposition.tsvd import TruncatedSVD
from cuml.ensemble.randomforestclassifier import RandomForestClassifier
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml.explainer.kernel_shap import KernelExplainer
from cuml.explainer.permutation_shap import PermutationExplainer
from cuml.explainer.tree_shap import TreeExplainer
from cuml.fil import ForestInference, fil
from cuml.internals.base import Base
from cuml.internals.global_settings import (
    GlobalSettings,
    _global_settings_data,
)
from cuml.internals.memory_utils import (
    set_global_output_type,
    using_output_type,
)
from cuml.kernel_ridge.kernel_ridge import KernelRidge
from cuml.linear_model.elastic_net import ElasticNet
from cuml.linear_model.lasso import Lasso
from cuml.linear_model.linear_regression import LinearRegression
from cuml.linear_model.logistic_regression import LogisticRegression
from cuml.linear_model.mbsgd_classifier import MBSGDClassifier
from cuml.linear_model.mbsgd_regressor import MBSGDRegressor
from cuml.linear_model.ridge import Ridge
from cuml.manifold.t_sne import TSNE
from cuml.manifold.umap import UMAP
from cuml.metrics import accuracy_score, adjusted_rand_score, r2_score
from cuml.model_selection import train_test_split
from cuml.naive_bayes.naive_bayes import MultinomialNB
from cuml.neighbors.kernel_density import KernelDensity
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from cuml.preprocessing.LabelEncoder import LabelEncoder
from cuml.random_projection.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from cuml.solvers.cd import CD
from cuml.solvers.qn import QN
from cuml.solvers.sgd import SGD
from cuml.svm import SVC, SVR, LinearSVC, LinearSVR
from cuml.tsa import stationarity
from cuml.tsa.arima import ARIMA
from cuml.tsa.auto_arima import AutoARIMA
from cuml.tsa.holtwinters import ExponentialSmoothing


def __getattr__(name):

    if name == "global_settings":
        try:
            return _global_settings_data.settings
        except AttributeError:
            _global_settings_data.settings = GlobalSettings()
            return _global_settings_data.settings

    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    # Modules
    "common",
    "feature_extraction",
    "fil",
    "metrics",
    "multiclass",
    "naive_bayes",
    "preprocessing",
    "explainer",
    # Classes
    "AgglomerativeClustering",
    "ARIMA",
    "AutoARIMA",
    "Base",
    "CD",
    "DBSCAN",
    "ElasticNet",
    "ExponentialSmoothing",
    "ForestInference",
    "GaussianRandomProjection",
    "Handle",
    "HDBSCAN",
    "IncrementalPCA",
    "KernelDensity",
    "KernelExplainer",
    "KernelRidge",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "Lasso",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "MBSGDClassifier",
    "MBSGDRegressor",
    "NearestNeighbors",
    "PCA",
    "PermutationExplainer",
    "QN",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "SGD",
    "SparseRandomProjection",
    "SVC",
    "SVR",
    "TruncatedSVD",
    "TreeExplainer",
    "TSNE",
    "UMAP",
    # Functions
    "johnson_lindenstrauss_min_dim",
    "make_arima",
    "make_blobs",
    "make_classification",
    "make_regression",
    "stationarity",
]
