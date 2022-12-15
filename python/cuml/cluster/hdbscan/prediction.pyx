# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from cython.operator cimport dereference as deref

import numpy as np
import cupy as cp

from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.common.doc_utils import generate_docstring
from pylibraft.common.handle cimport handle_t

from pylibraft.common.handle import Handle
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.mixins import ClusterMixin
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals import logger
from cuml.internals.import_utils import has_hdbscan_plots

import cuml
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML::HDBSCAN::Common":

    cdef cppclass CondensedHierarchy[value_idx, value_t]:
        CondensedHierarchy(
            const handle_t &handle, size_t n_leaves)

        value_idx *get_parents()
        value_idx *get_children()
        value_t *get_lambdas()
        value_idx get_n_edges()

    cdef cppclass hdbscan_output[int, float]:
        hdbscan_output(const handle_t &handle,
                       int n_leaves,
                       int *labels,
                       float *probabilities,
                       int *children,
                       int *sizes,
                       float *deltas,
                       int *mst_src,
                       int *mst_dst,
                       float *mst_weights)

        int get_n_leaves()
        int get_n_clusters()
        float *get_stabilities()
        int *get_labels()
        CondensedHierarchy[int, float] &get_condensed_tree()

    cdef cppclass PredictionData[int, float]:
        PredictionData(const handle_t &handle,
                       int m,
                       int n)

        size_t n_rows
        size_t n_cols

cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML":

    void compute_all_points_membership_vectors(
        const handle_t &handle,
        CondensedHierarchy[int, float] &condensed_tree,
        PredictionData[int, float] &prediction_data_,
        float* X,
        DistanceType metric,
        float* membership_vec)

    void out_of_sample_predict(const handle_t &handle,
                               CondensedHierarchy[int, float] &condensed_tree,
                               PredictionData[int, float] &prediction_data,
                               float* X,
                               int* labels,
                               float* points_to_predict,
                               size_t n_prediction_points,
                               DistanceType metric,
                               int min_samples,
                               int* out_labels,
                               float* out_probabilities)

_metrics_mapping = {
    'l1': DistanceType.L1,
    'cityblock': DistanceType.L1,
    'manhattan': DistanceType.L1,
    'l2': DistanceType.L2SqrtExpanded,
    'euclidean': DistanceType.L2SqrtExpanded,
    'cosine': DistanceType.CosineExpanded
}


def all_points_membership_vectors(clusterer):

    """
    Predict soft cluster membership vectors for all points in the
    original dataset the clusterer was trained on. This function is more
    efficient by making use of the fact that all points are already in the
    condensed tree, and processing in bulk.

    Parameters
    ----------
    clusterer : HDBSCAN
         A clustering object that has been fit to the data and
        had ``prediction_data=True`` set.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` of the original dataset is a member of
        cluster ``j`` is in ``membership_vectors[i, j]``.
    """

    if not clusterer.fit_called_:
        raise ValueError("The clusterer is not fit on data. "
                         "Please call clusterer.fit first")

    if not clusterer.prediction_data:
        raise ValueError("PredictionData not generated. "
                         "Please call clusterer.fit again with "
                         "prediction_data=True")

    if clusterer.n_clusters_ == 0:
        return np.zeros(clusterer.n_rows, dtype=np.float32)

    cdef uintptr_t input_ptr = clusterer.X_m.ptr

    membership_vec = CumlArray.empty(
        (clusterer.n_rows * clusterer.n_clusters_,),
        dtype="float32")

    cdef uintptr_t membership_vec_ptr = membership_vec.ptr

    cdef hdbscan_output *hdbscan_output_ = \
        <hdbscan_output*><size_t>clusterer.hdbscan_output_

    cdef PredictionData *prediction_data_ = \
        <PredictionData*><size_t>clusterer.prediction_data_ptr

    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()
    compute_all_points_membership_vectors(handle_[0],
                                          hdbscan_output_.get_condensed_tree(),
                                          deref(prediction_data_),
                                          <float*> input_ptr,
                                          _metrics_mapping[clusterer.metric],
                                          <float*> membership_vec_ptr)

    clusterer.handle.sync()
    return membership_vec.to_output(
        output_type="numpy",
        output_dtype="float32").reshape((clusterer.n_rows,
                                         clusterer.n_clusters_))


def approximate_predict(clusterer, points_to_predict, convert_dtype=True):
    """Predict the cluster label of new points. The returned labels
    will be those of the original clustering found by ``clusterer``,
    and therefore are not (necessarily) the cluster labels that would
    be found by clustering the original data combined with
    ``points_to_predict``, hence the 'approximate' label.

    If you simply wish to assign new points to an existing clustering
    in the 'best' way possible, this is the function to use. If you
    want to predict how ``points_to_predict`` would cluster with
    the original data under HDBSCAN the most efficient existing approach
    is to simply recluster with the new point(s) added to the original dataset.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        had ``prediction_data=True`` set.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    Returns
    -------
    labels : array (n_samples,)
        The predicted labels of the ``points_to_predict``

    probabilities : array (n_samples,)
        The soft cluster scores for each of the ``points_to_predict``
    """

    if not clusterer.fit_called_:
        raise ValueError("The clusterer is not fit on data. "
                         "Please call clusterer.fit first")

    if not clusterer.prediction_data:
        raise ValueError("PredictionData not generated. "
                         "Please call clusterer.fit again with "
                         "prediction_data=True")

    if clusterer.n_clusters_ == 0:
        logger.warn(
            'Clusterer does not have any defined clusters, new data '
            'will be automatically predicted as outliers.'
        )

    points_to_predict_m, n_prediction_points, n_cols, _ = \
        input_to_cuml_array(points_to_predict, order='C',
                            check_dtype=[np.float32],
                            convert_to_dtype=(np.float32
                                              if convert_dtype
                                              else None))

    if n_cols != clusterer.n_cols:
        raise ValueError('New points dimension does not match fit data!')

    cdef uintptr_t prediction_ptr = points_to_predict_m.ptr
    cdef uintptr_t input_ptr = clusterer.X_m.ptr

    prediction_labels = CumlArray.empty(
        (n_prediction_points,),
        dtype="int32")

    cdef uintptr_t prediction_labels_ptr = prediction_labels.ptr

    prediction_probs = CumlArray.empty(
        (n_prediction_points,),
        dtype="float32")

    cdef uintptr_t prediction_probs_ptr = prediction_probs.ptr

    cdef hdbscan_output *hdbscan_output_ = \
        <hdbscan_output*><size_t>clusterer.hdbscan_output_

    cdef PredictionData *prediction_data_ = \
        <PredictionData*><size_t>clusterer.prediction_data_ptr

    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    out_of_sample_predict(handle_[0],
                          hdbscan_output_.get_condensed_tree(),
                          deref(prediction_data_),
                          <float*> input_ptr,
                          <int*> hdbscan_output_.get_labels(),
                          <float*> prediction_ptr,
                          n_prediction_points,
                          _metrics_mapping[clusterer.metric],
                          clusterer.min_samples,
                          <int*> prediction_labels_ptr,
                          <float*> prediction_probs_ptr)

    clusterer.handle.sync()
    return prediction_labels.to_output(output_type="numpy"),\
        prediction_probs.to_output(output_type="numpy", output_dtype="float32")
