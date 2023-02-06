#
# Copyright (c) 2022, NVIDIA CORPORATION.
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


from cuml.internals.base import Base, UniversalBase
# breakpoint()

# GPU only packages

import cuml.common.cuda as cuda
from cuml.common.handle import Handle
# breakpoint()

from cuml.cluster.dbscan import DBSCAN
# breakpoint()
from cuml.cluster.kmeans import KMeans
# breakpoint()
from cuml.cluster.agglomerative import AgglomerativeClustering
# breakpoint()
from cuml.cluster.hdbscan import HDBSCAN
# breakpoint()

from cuml.datasets.arima import make_arima
# breakpoint()
from cuml.datasets.blobs import make_blobs
# breakpoint()
from cuml.datasets.regression import make_regression
# breakpoint()
from cuml.datasets.classification import make_classification
# breakpoint()

from cuml.decomposition.pca import PCA
# breakpoint()
from cuml.decomposition.tsvd import TruncatedSVD
# breakpoint()
from cuml.decomposition.incremental_pca import IncrementalPCA
# breakpoint()

from cuml.fil.fil import ForestInference
# breakpoint()

from cuml.ensemble.randomforestclassifier import RandomForestClassifier
# breakpoint()
from cuml.ensemble.randomforestregressor import RandomForestRegressor
# breakpoint()

from cuml.explainer.kernel_shap import KernelExplainer
# breakpoint()
from cuml.explainer.permutation_shap import PermutationExplainer
# breakpoint()
from cuml.explainer.tree_shap import TreeExplainer
# breakpoint()

import cuml.feature_extraction
from cuml.fil import fil
# breakpoint()

from cuml.internals.global_settings import (
    GlobalSettings, _global_settings_data)
# breakpoint()

from cuml.kernel_ridge.kernel_ridge import KernelRidge
# breakpoint()

from cuml.linear_model.elastic_net import ElasticNet
# breakpoint()
from cuml.linear_model.lasso import Lasso
# breakpoint()
from cuml.linear_model.logistic_regression import LogisticRegression
# breakpoint()
from cuml.linear_model.mbsgd_classifier import MBSGDClassifier
# breakpoint()
from cuml.linear_model.mbsgd_regressor import MBSGDRegressor
# breakpoint()
from cuml.linear_model.ridge import Ridge
# breakpoint()

from cuml.manifold.t_sne import TSNE
# breakpoint()
from cuml.manifold.umap import UMAP
# breakpoint()
from cuml.metrics.accuracy import accuracy_score
# breakpoint()
from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
# breakpoint()
from cuml.metrics.regression import r2_score
# breakpoint()
from cuml.model_selection import train_test_split
# breakpoint()

from cuml.naive_bayes.naive_bayes import MultinomialNB
# breakpoint()

from cuml.neighbors.nearest_neighbors import NearestNeighbors
# breakpoint()
from cuml.neighbors.kernel_density import KernelDensity
# breakpoint()
from cuml.neighbors.kneighbors_classifier import KNeighborsClassifier
# breakpoint()
from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor
# breakpoint()

from cuml.preprocessing.LabelEncoder import LabelEncoder
# breakpoint()

from cuml.random_projection.random_projection import \
    GaussianRandomProjection
from cuml.random_projection.random_projection import SparseRandomProjection
# breakpoint()
from cuml.random_projection.random_projection import \
    johnson_lindenstrauss_min_dim

from cuml.solvers.cd import CD
# breakpoint()
from cuml.solvers.sgd import SGD
# breakpoint()
from cuml.solvers.qn import QN
# breakpoint()
from cuml.svm import SVC
# breakpoint()
from cuml.svm import SVR
# breakpoint()
from cuml.svm import LinearSVC
# breakpoint()
from cuml.svm import LinearSVR
# breakpoint()

from cuml.tsa import stationarity
# breakpoint()
from cuml.tsa.arima import ARIMA
# breakpoint()
from cuml.tsa.auto_arima import AutoARIMA
# breakpoint()
from cuml.tsa.holtwinters import ExponentialSmoothing
# breakpoint()

from cuml.common.pointer_utils import device_of_gpu_matrix
# breakpoint()
from cuml.internals.memory_utils import (
    set_global_output_type, using_output_type
)

# Universal packages

from cuml.linear_model.linear_regression import LinearRegression
# breakpoint()

# Import verion. Remove at end of file
from ._version import get_versions
# breakpoint()


# Version configuration
__version__ = get_versions()['version']
del get_versions


def __getattr__(name):

    if name == 'global_settings':
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
    "cuda",
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
    "UniversalBase",
    # Functions
    "johnson_lindenstrauss_min_dim",
    "make_arima",
    "make_blobs",
    "make_classification",
    "make_regression",
    "stationarity",
]
