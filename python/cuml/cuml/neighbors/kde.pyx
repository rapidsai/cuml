#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.base import get_handle

from libc.stdint cimport int64_t, uintptr_t
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType
from cuml.metrics.pairwise_distances import PAIRWISE_DISTANCE_METRICS


# All types and the wrapper function come from the single cuML header,
# without requiring cuVS headers in the generated extension target.
# handle_t extends raft::device_resources (raft::resources), so passing
# handle_t to a raft::resources const& parameter is valid.
# The alias _cuml_kde_score_samples avoids shadowing the Python def below.
cdef extern from "cuml/neighbors/kde.hpp" nogil:

    ctypedef enum class DensityKernelType "ML::KDE::DensityKernelType":
        Gaussian "ML::KDE::DensityKernelType::Gaussian"
        Tophat "ML::KDE::DensityKernelType::Tophat"
        Epanechnikov "ML::KDE::DensityKernelType::Epanechnikov"
        Exponential "ML::KDE::DensityKernelType::Exponential"
        Linear "ML::KDE::DensityKernelType::Linear"
        Cosine "ML::KDE::DensityKernelType::Cosine"

    void _cuml_kde_score_samples \
        "ML::KDE::score_samples"(const handle_t &handle,
                                 const float *query,
                                 const float *train,
                                 const float *weights,
                                 float *output,
                                 int64_t n_query,
                                 int64_t n_train,
                                 int64_t n_features,
                                 float bandwidth,
                                 float sum_weights,
                                 DensityKernelType kernel,
                                 DistanceType metric,
                                 float metric_arg) except +

    void _cuml_kde_score_samples \
        "ML::KDE::score_samples"(const handle_t &handle,
                                 const double *query,
                                 const double *train,
                                 const double *weights,
                                 double *output,
                                 int64_t n_query,
                                 int64_t n_train,
                                 int64_t n_features,
                                 double bandwidth,
                                 double sum_weights,
                                 DensityKernelType kernel,
                                 DistanceType metric,
                                 double metric_arg) except +


KDE_KERNEL_TYPES = {
    "gaussian": DensityKernelType.Gaussian,
    "tophat": DensityKernelType.Tophat,
    "epanechnikov": DensityKernelType.Epanechnikov,
    "exponential": DensityKernelType.Exponential,
    "linear": DensityKernelType.Linear,
    "cosine": DensityKernelType.Cosine,
}

VALID_KERNELS = list(KDE_KERNEL_TYPES.keys())


def kde_score_samples(query, train, sample_weight,
                      bandwidth, sum_weights,
                      kernel_str, metric_str, metric_arg=2.0):
    """Compute log-density via fused cuVS KDE kernel.

    Parameters
    ----------
    query : CumlArray, shape (n_query, d)
    train : CuPy array, shape (n_train, d)
    sample_weight : CuPy array (n_train,) or None
    bandwidth : float
    sum_weights : float
    kernel_str : str
    metric_str : str
    metric_arg : float

    Returns
    -------
    CumlArray of shape (n_query,) — final normalized log-density
    """
    if kernel_str not in KDE_KERNEL_TYPES:
        raise ValueError(f"kernel={kernel_str!r} is not supported")
    if metric_str not in PAIRWISE_DISTANCE_METRICS:
        raise ValueError(f"metric={metric_str!r} is not supported")

    cdef DensityKernelType kernel_enum = KDE_KERNEL_TYPES[kernel_str]
    cdef DistanceType metric_enum = PAIRWISE_DISTANCE_METRICS[metric_str]

    # Determine dtype from training data
    dtype = train.dtype

    cdef int64_t n_query = query.shape[0]
    cdef int64_t n_train = train.shape[0]
    cdef int64_t n_features = query.shape[1] if len(query.shape) > 1 else 1

    # Allocate output
    output = CumlArray.zeros(n_query, dtype=dtype)

    # Extract device pointers
    cdef uintptr_t query_ptr = query.__cuda_array_interface__['data'][0]
    cdef uintptr_t train_ptr = train.__cuda_array_interface__['data'][0]
    cdef uintptr_t weight_ptr = 0
    if sample_weight is not None:
        weight_ptr = sample_weight.__cuda_array_interface__['data'][0]
    cdef uintptr_t output_ptr = output.ptr

    # Precompute typed weight pointers outside nogil
    cdef const float* weights_f = (
        <const float*>weight_ptr if weight_ptr != 0
        else <const float*>NULL
    )
    cdef const double* weights_d = (
        <const double*>weight_ptr if weight_ptr != 0
        else <const double*>NULL
    )

    # Convert Python scalars to C types before entering nogil
    cdef double c_bandwidth = <double>bandwidth
    cdef double c_sum_weights = <double>sum_weights
    cdef double c_metric_arg = <double>metric_arg

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

    if dtype == np.float32:
        with nogil:
            _cuml_kde_score_samples(
                handle_[0],
                <const float*>query_ptr,
                <const float*>train_ptr,
                weights_f,
                <float*>output_ptr,
                n_query, n_train, n_features,
                <float>c_bandwidth,
                <float>c_sum_weights,
                kernel_enum,
                metric_enum,
                <float>c_metric_arg)
    elif dtype == np.float64:
        with nogil:
            _cuml_kde_score_samples(
                handle_[0],
                <const double*>query_ptr,
                <const double*>train_ptr,
                weights_d,
                <double*>output_ptr,
                n_query, n_train, n_features,
                c_bandwidth,
                c_sum_weights,
                kernel_enum,
                metric_enum,
                c_metric_arg)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

    handle.sync()

    return output
