#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cuml.common.base import Base
from cuml.common.handle import Handle
import cuml.common.cuda as cuda

from cuml.cluster.dbscan import DBSCAN
from cuml.cluster.kmeans import KMeans

from cuml.datasets.arima import make_arima
from cuml.datasets.blobs import make_blobs
from cuml.datasets.regression import make_regression
from cuml.datasets.classification import make_classification

from cuml.decomposition.pca import PCA
from cuml.decomposition.tsvd import TruncatedSVD

from cuml.fil.fil import ForestInference

from cuml.ensemble.randomforestclassifier import RandomForestClassifier
from cuml.ensemble.randomforestregressor import RandomForestRegressor

from cuml.fil import fil

from cuml.linear_model.elastic_net import ElasticNet
from cuml.linear_model.lasso import Lasso
from cuml.linear_model.linear_regression import LinearRegression
from cuml.linear_model.logistic_regression import LogisticRegression
from cuml.linear_model.mbsgd_classifier import MBSGDClassifier
from cuml.linear_model.mbsgd_regressor import MBSGDRegressor
from cuml.linear_model.ridge import Ridge

from cuml.manifold.t_sne import TSNE
from cuml.manifold.umap import UMAP
from cuml.metrics.accuracy import accuracy_score
from cuml.metrics.cluster.adjustedrandindex import adjusted_rand_score
from cuml.metrics.regression import r2_score

from cuml.naive_bayes.naive_bayes import MultinomialNB

from cuml.neighbors.nearest_neighbors import NearestNeighbors

from cuml.preprocessing.LabelEncoder import LabelEncoder
from cuml.preprocessing.model_selection import train_test_split

from cuml.random_projection.random_projection import GaussianRandomProjection
from cuml.random_projection.random_projection import SparseRandomProjection
from cuml.random_projection.random_projection import \
    johnson_lindenstrauss_min_dim

from cuml.solvers.cd import CD
from cuml.solvers.sgd import SGD
from cuml.solvers.qn import QN
from cuml.svm import SVC
from cuml.svm import SVR

from cuml.tsa import stationarity
from cuml.tsa.arima import ARIMA
from cuml.tsa.auto_arima import AutoARIMA
from cuml.tsa.holtwinters import ExponentialSmoothing

from cuml.common.pointer_utils import device_of_gpu_matrix

# RAFT

from cuml.raft import raft_include_test


# Output type configuration

global_output_type = 'input'

from cuml.common.memory_utils import set_global_output_type, using_output_type

# Version configuration

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

