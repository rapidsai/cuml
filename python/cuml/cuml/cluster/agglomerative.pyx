#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base, get_handle
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin
from cuml.internals.outputs import reflect

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/linkage.hpp" namespace "ML::linkage" nogil:

    cdef void single_linkage(
        const handle_t &handle,
        const float *X,
        int n_rows,
        int n_cols,
        size_t n_clusters,
        DistanceType metric,
        int* children,
        int* labels,
        bool use_knn,
        int c,
    ) except +


_metrics_mapping = {
    "l1": DistanceType.L1,
    "cityblock": DistanceType.L1,
    "manhattan": DistanceType.L1,
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "cosine": DistanceType.CosineExpanded
}


class AgglomerativeClustering(Base, ClusterMixin, CMajorInputTagMixin):
    """
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases a
    given linkage distance.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    metric : str, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1",
        "l2", "manhattan", or "cosine". If connectivity is "knn" only
        "euclidean" is accepted.
    connectivity : {"pairwise", "knn"}, default="knn"
        The type of connectivity matrix to compute.
         * 'pairwise' will compute the entire fully-connected graph of
           pairwise distances between each set of points. This is the
           fastest to compute and can be very fast for smaller datasets
           but requires O(n^2) space.
         * 'knn' will sparsify the fully-connected connectivity matrix to save
           memory and enable much larger inputs. You can use ``c`` to influence
           the number of neighbors used.
    linkage : {"single"}, default="single"
        Which linkage criterion to use. The linkage criterion determines
        which distance to use between sets of observations. The algorithm
        will merge the pairs of clusters that minimize this criterion.

         * 'single' uses the minimum of the distances between all
           observations of the two sets.
    c : int, default=15
        Indirectly influences the number of neighbors to use when
        ``connectivity="knn"``, with ``n_neighbors = log(n_samples) + c``. The
        default of 15 should suffice for most problems.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels : array, shape (n_samples,)
        Cluster labels for each point.
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_connected_components_ : int
        The estimated number of connected components in the graph.
    children_ : array, shape (n_samples - 1, 2)
        The children of each non-leave node.
    """

    labels_ = CumlArrayDescriptor(order="C")
    children_ = CumlArrayDescriptor(order="C")

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_clusters",
            "metric",
            "linkage",
            "connectivity",
            "c",
        ]

    def __init__(
        self,
        n_clusters=2,
        *,
        metric="euclidean",
        connectivity="knn",
        linkage="single",
        c=15,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)

        self.n_clusters = n_clusters
        self.metric = metric
        self.connectivity = connectivity
        self.linkage = linkage
        self.c = c

    @generate_docstring()
    @reflect(reset=True)
    def fit(self, X, y=None, *, convert_dtype=True) -> "AgglomerativeClustering":
        """
        Fit the hierarchical clustering from features.
        """
        # Validate and process inputs
        X = input_to_cuml_array(
            X,
            order="C",
            check_dtype=np.float32,
            convert_to_dtype=(np.float32 if convert_dtype else None),
        ).array
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        if n_rows < 2:
            raise ValueError(
                f"Found array with {n_rows} sample(s) (shape={X.shape}) while a "
                f"minimum of 2 is required."
            )
        if n_cols < 1:
            raise ValueError(
                f"Found array with {n_cols} feature(s) (shape={X.shape}) while "
                f"a minimum of 1 is required."
            )

        # Validate and process hyperparameters
        if self.linkage != "single":
            raise ValueError("Only single linkage clustering is supported currently")

        if self.connectivity not in ["knn", "pairwise"]:
            raise ValueError("'connectivity' can only be one of {'knn', 'pairwise'}")
        cdef bool use_knn = self.connectivity == "knn"

        if self.metric not in _metrics_mapping:
            raise ValueError("metric={self.metric!r} not supported")
        cdef DistanceType metric = _metrics_mapping[self.metric]

        cdef size_t n_clusters = self.n_clusters
        if n_clusters < 1 or n_clusters > n_rows:
            raise ValueError(
                f"Expected 1 <= n_clusters <= n_rows ({n_rows}), got {n_clusters=}"
            )

        # Allocate outputs
        labels = CumlArray.empty(n_rows, dtype="int32", order="C")
        children = CumlArray.empty((n_rows - 1, 2), dtype="int32", order="C")

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef int c = self.c
        cdef float* X_ptr = <float*><uintptr_t>X.ptr
        cdef int* children_ptr = <int*><uintptr_t>children.ptr
        cdef int* labels_ptr = <int*><uintptr_t>labels.ptr

        # Perform fit
        with nogil:
            single_linkage(
                handle_[0],
                X_ptr,
                n_rows,
                n_cols,
                n_clusters,
                metric,
                children_ptr,
                labels_ptr,
                use_knn,
                c,
            )
        handle.sync()

        # We only support single linkage for now, for other linkage types
        # n_connected_components_ and n_leaves_ will differ
        self.n_connected_components_ = 1
        self.n_leaves_ = n_rows
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.children_ = children

        return self

    @generate_docstring(return_values={"name": "preds",
                                       "type": "dense",
                                       "description": "Cluster indexes",
                                       "shape": "(n_samples, 1)"})
    @reflect
    def fit_predict(self, X, y=None) -> CumlArray:
        """
        Fit and return the assigned cluster labels.
        """
        return self.fit(X).labels_
