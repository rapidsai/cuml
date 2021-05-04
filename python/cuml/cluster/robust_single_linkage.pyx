#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

# distutils: language = c++

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from cython.operator cimport dereference as deref

import numpy as np
import cupy as cp

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.mixins import ClusterMixin
from cuml.common.mixins import CMajorInputTagMixin


from cuml.metrics.distance_type cimport DistanceType

from .hdbscan_plot import SingleLinkageTree

class RobustSingleLinkage(Base, ClusterMixin, CMajorInputTagMixin):
    """Perform robust single linkage clustering from a vector array
    or distance matrix.
    Robust single linkage is a modified version of single linkage that
    attempts to be more robust to noise. Specifically the goal is to
    more accurately approximate the level set tree of the unknown
    probability density function from which the sample data has
    been drawn.

    Parameters
    ----------
    X : array of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array
    cut : float
        The reachability distance value to cut the cluster heirarchy at
        to derive a flat cluster labelling.
    k : int, optional (default=5)
        Reachability distances will be computed with regard to the `k`
        nearest neighbors.
    alpha : float, optional (default=np.sqrt(2))
        Distance scaling for reachability distance computation. Reachability
        distance is computed as
        $max \{ core_k(a), core_k(b), 1/\alpha d(a,b) \}$.
    gamma : int, optional (default=5)
        Ignore any clusters in the flat clustering with size less than gamma,
        and declare points in such clusters as noise points.
    metric : string, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array.
    metric_params : dict, option (default={})
        Keyword parameter arguments for calling the metric (for example
        the p values if using the minkowski metric).

    Attributes
    -------
    labels_ : ndarray, shape (n_samples, )
        Cluster labels for each point.  Noisy samples are given the label -1.
    cluster_hierarchy_ : SingleLinkageTree object
        The single linkage tree produced during clustering.
        This object provides several methods for:
            * Plotting
            * Generating a flat clustering
            * Exporting to NetworkX
            * Exporting to Pandas

    References
    ----------
    .. [1] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    """

    def __init__(self, cut=0.4, k=5, alpha=1.4142135623730951, gamma=5,
                 metric='euclidean', algorithm='best', core_dist_n_jobs=4,
                 metric_params={}):

        self.cut = cut
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):

        # TODO: Construct SingleLinkageTree
        return self