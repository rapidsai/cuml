#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base

from pylibraft.common.handle cimport handle_t

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/single_linkage_output.hpp" namespace "ML" nogil:

    cdef cppclass single_linkage_output[int]:
        int m
        int n_clusters
        int n_leaves
        int n_connected_components
        int *labels
        int *children

cdef extern from "cuml/cluster/linkage.hpp" namespace "ML" nogil:

    cdef void single_linkage_pairwise(
        const handle_t &handle,
        const float *X,
        size_t m,
        size_t n,
        single_linkage_output[int] *out,
        DistanceType metric,
        int n_clusters
    ) except +

    cdef void single_linkage_neighbors(
        const handle_t &handle,
        const float *X,
        size_t m,
        size_t n,
        single_linkage_output[int] *out,
        DistanceType metric,
        int c,
        int n_clusters
    ) except +


_metrics_mapping = {
    'l1': DistanceType.L1,
    'cityblock': DistanceType.L1,
    'manhattan': DistanceType.L1,
    'l2': DistanceType.L2SqrtExpanded,
    'euclidean': DistanceType.L2SqrtExpanded,
    'cosine': DistanceType.CosineExpanded
}


class AgglomerativeClustering(Base, ClusterMixin, CMajorInputTagMixin):

    """
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases a
    given linkage distance.

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    n_clusters : int (default = 2)
        The number of clusters to find.
    metric : str, default=None
        Metric used to compute the linkage. Can be "euclidean", "l1",
        "l2", "manhattan", or "cosine". If set to `None` then "euclidean"
        is used. If connectivity is "knn" only "euclidean" is accepted.
        .. versionadded:: 24.06

    linkage : {"single"}, default="single"
        Which linkage criterion to use. The linkage criterion determines
        which distance to use between sets of observations. The algorithm
        will merge the pairs of clusters that minimize this criterion.

         * 'single' uses the minimum of the distances between all
           observations of the two sets.

    n_neighbors : int (default = 15)
        The number of neighbors to compute when connectivity = "knn"
    connectivity : {"pairwise", "knn"}, (default = "knn")
        The type of connectivity matrix to compute.
         * 'pairwise' will compute the entire fully-connected graph of
           pairwise distances between each set of points. This is the
           fastest to compute and can be very fast for smaller datasets
           but requires O(n^2) space.
         * 'knn' will sparsify the fully-connected connectivity matrix to
           save memory and enable much larger inputs. "n_neighbors" will
           control the amount of memory used and the graph will be connected
           automatically in the event "n_neighbors" was not large enough
           to connect it.

    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    """

    labels_ = CumlArrayDescriptor()
    children_ = CumlArrayDescriptor()

    def __init__(self, *, n_clusters=2, metric=None,
                 linkage="single", handle=None, verbose=False,
                 connectivity='knn', n_neighbors=10, output_type=None):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        if linkage is not "single":
            raise ValueError("Only single linkage clustering is "
                             "supported currently")

        if connectivity not in ["knn", "pairwise"]:
            raise ValueError("'connectivity' can only be one of "
                             "{'knn', 'pairwise'}")

        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be >= 1")

        if n_neighbors > 1023 or n_neighbors < 2:
            raise ValueError("'n_neighbors' must be a positive number "
                             "between 2 and 1023")

        if metric is not None and metric not in _metrics_mapping:
            raise ValueError(f"Metric {metric!r} is not supported.")

        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.n_neighbors = n_neighbors
        self.connectivity = connectivity

        self.n_clusters_ = None
        self.n_leaves_ = None
        self.n_connected_components_ = None
        self.distances_ = None

    @generate_docstring()
    def fit(self, X, y=None, *, convert_dtype=True) -> "AgglomerativeClustering":
        """
        Fit the hierarchical clustering from features.
        """
        if self.metric is None:
            metric_name = "euclidean"
        else:
            metric_name = self.metric

        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='C',
                                check_dtype=[np.float32],
                                convert_to_dtype=(np.float32
                                                  if convert_dtype
                                                  else None))

        if self.n_clusters > n_rows:
            raise ValueError("'n_clusters' must be <= n_samples")

        cdef uintptr_t input_ptr = X_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Hardcode n_components_ to 1 for single linkage. This will
        # not be the case for other linkage types.
        self.n_connected_components_ = 1
        self.n_leaves_ = n_rows
        self.n_clusters_ = self.n_clusters

        self.labels_ = CumlArray.empty(n_rows, dtype="int32")
        self.children_ = CumlArray.empty((2, n_rows), dtype="int32")
        cdef uintptr_t labels_ptr = self.labels_.ptr
        cdef uintptr_t children_ptr = self.children_.ptr

        cdef single_linkage_output[int] linkage_output
        linkage_output.children = <int*>children_ptr
        linkage_output.labels = <int*>labels_ptr

        cdef DistanceType metric
        if metric_name in _metrics_mapping:
            metric = _metrics_mapping[metric_name]
        else:
            raise ValueError("Metric '%s' not supported." % metric_name)

        if self.connectivity == 'knn':
            single_linkage_neighbors(
                handle_[0], <float*>input_ptr, <int> n_rows,
                <int> n_cols, <single_linkage_output[int]*> &linkage_output,
                <DistanceType> metric, <int>self.n_neighbors,
                <int> self.n_clusters)
        elif self.connectivity == 'pairwise':
            single_linkage_pairwise(
                handle_[0], <float*>input_ptr, <int> n_rows,
                <int> n_cols, <single_linkage_output[int]*> &linkage_output,
                <DistanceType> metric, <int> self.n_clusters)
        else:
            raise ValueError("'connectivity' can only be one of "
                             "{'knn', 'pairwise'}")

        self.handle.sync()

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    def fit_predict(self, X, y=None) -> CumlArray:
        """
        Fit the hierarchical clustering from features and return
        cluster labels.
        """
        return self.fit(X).labels_

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_clusters",
            "metric",
            "linkage",
            "connectivity",
            "n_neighbors"
        ]
