# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')
from cuml.internals.safe_imports import gpu_only_import
cp = gpu_only_import('cupy')

from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.common.doc_utils import generate_docstring
from pylibraft.common.handle cimport handle_t

from pylibraft.common.handle import Handle
from cuml.common import (
    input_to_cuml_array,
    input_to_host_array
)
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.available_devices import is_cuda_available
from cuml.internals.device_type import DeviceType
from cuml.internals.mixins import ClusterMixin
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals import logger
from cuml.internals.import_utils import has_hdbscan

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
        int *get_inverse_label_map()
        float *get_core_dists()
        CondensedHierarchy[int, float] &get_condensed_tree()

    cdef cppclass PredictionData[int, float]:
        PredictionData(const handle_t &handle,
                       int m,
                       int n,
                       float *core_dists)

        size_t n_rows
        size_t n_cols

cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML":

    void compute_all_points_membership_vectors(
        const handle_t &handle,
        CondensedHierarchy[int, float] &condensed_tree,
        PredictionData[int, float] &prediction_data_,
        float* X,
        DistanceType metric,
        float* membership_vec,
        size_t batch_size)
    
    void compute_membership_vector(
        const handle_t& handle,
        CondensedHierarchy[int, float] &condensed_tree,
        PredictionData[int, float] &prediction_data,
        float* X,
        float* points_to_predict,
        size_t n_prediction_points,
        int min_samples,
        DistanceType metric,
        float* membership_vec,
        size_t batch_size);

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


def all_points_membership_vectors(clusterer, batch_size=4096):

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

    batch_size : int, optional, default=min(4096, n_rows)
        Lowers memory requirement by computing distance-based membership
        in smaller batches of points in the training data. For example, a batch
        size of 1,000 computes distance based memberships for 1,000 points at a
        time. The default batch size is 4,096.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` of the original dataset is a member of
        cluster ``j`` is in ``membership_vectors[i, j]``.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    device_type = cuml.global_settings.device_type

    # cpu infer, cpu/gpu train
    if device_type == DeviceType.host:
        assert has_hdbscan(raise_if_unavailable=True)
        from hdbscan.prediction import all_points_membership_vectors \
            as cpu_all_points_membership_vectors

        # trained on gpu
        if not hasattr(clusterer, "_cpu_model"):
            # the reference HDBSCAN implementations uses @property
            # for attributes without setters available for them,
            # so they can't be transferred from the GPU model
            # to the CPU model
            raise ValueError("Inferring on CPU is not supported yet when the "
                             "model has been trained on GPU")

        # this took a long debugging session to figure out, but
        # this method on cpu does not work without this copy for some reason
        clusterer._cpu_model.prediction_data_.raw_data = \
            clusterer._cpu_model.prediction_data_.raw_data.copy()
        return cpu_all_points_membership_vectors(clusterer._cpu_model)

    elif device_type == DeviceType.device:
        # trained on cpu
        if hasattr(clusterer, "_cpu_model"):
            clusterer._prep_cpu_to_gpu_prediction()

    if not clusterer.fit_called_:
        raise ValueError("The clusterer is not fit on data. "
                         "Please call clusterer.fit first")

    if not clusterer.prediction_data:
        raise ValueError("PredictionData not generated. "
                         "Please call clusterer.fit again with "
                         "prediction_data=True or call "
                         "clusterer.generate_prediction_data()")

    if clusterer.n_clusters_ == 0:
        return np.zeros(clusterer.n_rows, dtype=np.float32)

    cdef uintptr_t input_ptr = clusterer.X_m.ptr

    membership_vec = CumlArray.empty(
        (clusterer.n_rows * clusterer.n_clusters_,),
        dtype="float32")

    cdef uintptr_t membership_vec_ptr = membership_vec.ptr

    cdef PredictionData *prediction_data_ = \
        <PredictionData*><size_t>clusterer.prediction_data_ptr

    cdef CondensedHierarchy[int, float] *condensed_tree = \
        <CondensedHierarchy[int, float]*><size_t> clusterer.condensed_tree_ptr

    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    compute_all_points_membership_vectors(handle_[0],
                                          deref(condensed_tree),
                                          deref(prediction_data_),
                                          <float*> input_ptr,
                                          _metrics_mapping[clusterer.metric],
                                          <float*> membership_vec_ptr,
                                          batch_size)

    clusterer.handle.sync()
    return membership_vec.to_output(
        output_type="numpy",
        output_dtype="float32").reshape((clusterer.n_rows,
                                         clusterer.n_clusters_))


def membership_vector(clusterer, points_to_predict, batch_size=4096, convert_dtype=True):
    """Predict soft cluster membership. The result produces a vector
    for each point in ``points_to_predict`` that gives a probability that
    the given point is a member of a cluster for each of the selected clusters
    of the ``clusterer``.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        either had ``prediction_data=True`` set, or called the
        ``generate_prediction_data`` method after the fact.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.
    
    batch_size : int, optional, default=min(4096, n_points_to_predict)
        Lowers memory requirement by computing distance-based membership
        in smaller batches of points in the prediction data. For example, a
        batch size of 1,000 computes distance based memberships for 1,000
        points at a time. The default batch size is 4,096.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` is a member of cluster ``j`` is
        in ``membership_vectors[i, j]``.
    """

    device_type = cuml.global_settings.device_type

    # cpu infer, cpu/gpu train
    if device_type == DeviceType.host:
        assert has_hdbscan(raise_if_unavailable=True)
        from hdbscan.prediction import membership_vector \
            as cpu_membership_vector

        # trained on gpu
        if not hasattr(clusterer, "_cpu_model"):
            # the reference HDBSCAN implementations uses @property
            # for attributes without setters available for them,
            # so they can't be transferred from the GPU model
            # to the CPU model
            raise ValueError("Inferring on CPU is not supported yet when the "
                             "model has been trained on GPU")

        host_points_to_predict = input_to_host_array(points_to_predict).array
        return cpu_membership_vector(clusterer._cpu_model,
                                     host_points_to_predict)

    elif device_type == DeviceType.device:
        # trained on cpu
        if hasattr(clusterer, "_cpu_model"):
            clusterer._prep_cpu_to_gpu_prediction()

    if not clusterer.fit_called_:
        raise ValueError("The clusterer is not fit on data. "
                         "Please call clusterer.fit first")

    if not clusterer.prediction_data:
        raise ValueError("PredictionData not generated. "
                         "Please call clusterer.fit again with "
                         "prediction_data=True")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    points_to_predict_m, n_prediction_points, n_cols, _ = \
        input_to_cuml_array(points_to_predict, order='C',
                            check_dtype=[np.float32],
                            convert_to_dtype=(np.float32
                                              if convert_dtype
                                              else None))
    
    if clusterer.n_clusters_ == 0:
        return np.zeros(n_prediction_points, dtype=np.float32)

    if n_cols != clusterer.n_cols:
        raise ValueError('New points dimension does not match fit data!')

    cdef uintptr_t prediction_ptr = points_to_predict_m.ptr
    cdef uintptr_t input_ptr = clusterer.X_m.ptr
    
    membership_vec = CumlArray.empty(
        (n_prediction_points * clusterer.n_clusters_,),
        dtype="float32")

    cdef uintptr_t membership_vec_ptr = membership_vec.ptr

    cdef CondensedHierarchy[int, float] *condensed_tree = \
        <CondensedHierarchy[int, float]*><size_t> clusterer.condensed_tree_ptr

    cdef PredictionData *prediction_data_ = \
        <PredictionData*><size_t>clusterer.prediction_data_ptr

    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    compute_membership_vector(handle_[0],
                              deref(condensed_tree),
                              deref(prediction_data_),
                              <float*> input_ptr,
                              <float*> prediction_ptr,
                              n_prediction_points,
                              clusterer.min_samples,
                              _metrics_mapping[clusterer.metric],
                              <float*> membership_vec_ptr,
                              batch_size)

    clusterer.handle.sync()
    return membership_vec.to_output(
        output_type="numpy",
        output_dtype="float32").reshape((n_prediction_points,
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

    device_type = cuml.global_settings.device_type

    # cpu infer, cpu/gpu train
    if device_type == DeviceType.host:
        assert has_hdbscan(raise_if_unavailable=True)
        from hdbscan.prediction import approximate_predict \
            as cpu_approximate_predict

        # trained on gpu
        if not hasattr(clusterer, "_cpu_model"):
            # the reference HDBSCAN implementations uses @property
            # for attributes without setters available for them,
            # so they can't be transferred from the GPU model
            # to the CPU model
            raise ValueError("Inferring on CPU is not supported yet when the "
                             "model has been trained on GPU")

        host_points_to_predict = input_to_host_array(points_to_predict).array
        return cpu_approximate_predict(clusterer._cpu_model,
                                       host_points_to_predict)

    elif device_type == DeviceType.device:
        # trained on cpu
        if hasattr(clusterer, "_cpu_model"):
            clusterer._prep_cpu_to_gpu_prediction()

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

    cdef CondensedHierarchy[int, float] *condensed_tree = \
        <CondensedHierarchy[int, float]*><size_t> clusterer.condensed_tree_ptr

    cdef PredictionData *prediction_data_ = \
        <PredictionData*><size_t>clusterer.prediction_data_ptr

    labels, _, _, _ = input_to_cuml_array(clusterer.labels_,
                                          order="C",
                                          convert_to_dtype=np.int32)

    cdef uintptr_t labels_ptr = labels.ptr

    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    out_of_sample_predict(handle_[0],
                          deref(condensed_tree),
                          deref(prediction_data_),
                          <float*> input_ptr,
                          <int*> labels_ptr,
                          <float*> prediction_ptr,
                          n_prediction_points,
                          _metrics_mapping[clusterer.metric],
                          clusterer.min_samples,
                          <int*> prediction_labels_ptr,
                          <float*> prediction_probs_ptr)

    clusterer.handle.sync()
    return prediction_labels.to_output(output_type="numpy"),\
        prediction_probs.to_output(output_type="numpy", output_dtype="float32")
