#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cuml.metrics.trustworthiness import trustworthiness
from cuml.metrics.regression import r2_score
from cuml.metrics.regression import mean_squared_error
from cuml.metrics.regression import mean_squared_log_error
from cuml.metrics.regression import mean_absolute_error
from cuml.metrics.accuracy import accuracy_score
from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
from cuml.metrics._ranking import roc_auc_score
from cuml.metrics.precision_score import cython_precision_score as precision_score
from cuml.metrics._ranking import precision_recall_curve
from cuml.metrics._classification import log_loss
from cuml.metrics.cluster.homogeneity_score import \
    cython_homogeneity_score as homogeneity_score
from cuml.metrics.cluster.completeness_score import \
    cython_completeness_score as completeness_score
from cuml.metrics.cluster.mutual_info_score import \
    cython_mutual_info_score as mutual_info_score
from cuml.metrics.confusion_matrix import confusion_matrix
from cuml.metrics.cluster.entropy import cython_entropy as entropy
from cuml.metrics.pairwise_distances import pairwise_distances
from cuml.metrics.pairwise_distances import sparse_pairwise_distances
from cuml.metrics.pairwise_distances import PAIRWISE_DISTANCE_METRICS
from cuml.metrics.pairwise_distances import PAIRWISE_DISTANCE_SPARSE_METRICS
from cuml.metrics.pairwise_kernels import pairwise_kernels
from cuml.metrics.pairwise_kernels import PAIRWISE_KERNEL_FUNCTIONS
from cuml.metrics.hinge_loss import hinge_loss
from cuml.metrics.kl_divergence import kl_divergence
from cuml.metrics.cluster.v_measure import \
    cython_v_measure as v_measure_score

__all__ = [
    "trustworthiness",
    "r2_score",
    "mean_squared_error",
    "mean_squared_log_error",
    "mean_absolute_error",
    "accuracy_score",
    "adjusted_rand_score",
    "roc_auc_score",
    "precision_recall_curve",
    "log_loss",
    "homogeneity_score",
    "completeness_score",
    "mutual_info_score",
    "confusion_matrix",
    "entropy",
    "pairwise_distances",
    "sparse_pairwise_distances",
    "pairwise_kernels",
    "hinge_loss",
    "kl_divergence",
    "v_measure_score"
]
