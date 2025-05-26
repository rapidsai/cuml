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

import cupy as cp
import numpy as np

from cuml.common import input_to_cuml_array, using_output_type
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base, deprecate_non_keyword_only
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
    warn_legacy_device_interop,
)
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/dbscan.hpp" namespace "ML::Dbscan" nogil:

    ctypedef enum EpsNnMethod:
        BRUTE_FORCE "ML::Dbscan::EpsNnMethod::BRUTE_FORCE"
        RBC "ML::Dbscan::EpsNnMethod::RBC"

    cdef void fit(handle_t& handle,
                  float *input,
                  int n_rows,
                  int n_cols,
                  float eps,
                  int min_pts,
                  DistanceType metric,
                  int *labels,
                  int *core_sample_indices,
                  float* sample_weight,
                  size_t max_mbytes_per_batch,
                  EpsNnMethod eps_nn_method,
                  level_enum verbosity,
                  bool opg) except +

    cdef void fit(handle_t& handle,
                  double *input,
                  int n_rows,
                  int n_cols,
                  double eps,
                  int min_pts,
                  DistanceType metric,
                  int *labels,
                  int *core_sample_indices,
                  double* sample_weight,
                  size_t max_mbytes_per_batch,
                  EpsNnMethod eps_nn_method,
                  level_enum verbosity,
                  bool opg) except +

    cdef void fit(handle_t& handle,
                  float *input,
                  int64_t n_rows,
                  int64_t n_cols,
                  double eps,
                  int min_pts,
                  DistanceType metric,
                  int64_t *labels,
                  int64_t *core_sample_indices,
                  float* sample_weight,
                  size_t max_mbytes_per_batch,
                  EpsNnMethod eps_nn_method,
                  level_enum verbosity,
                  bool opg) except +

    cdef void fit(handle_t& handle,
                  double *input,
                  int64_t n_rows,
                  int64_t n_cols,
                  double eps,
                  int min_pts,
                  DistanceType metric,
                  int64_t *labels,
                  int64_t *core_sample_indices,
                  double* sample_weight,
                  size_t max_mbytes_per_batch,
                  EpsNnMethod eps_nn_method,
                  level_enum verbosity,
                  bool opg) except +


_SUPPORTED_METRICS = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "cosine": DistanceType.CosineExpanded,
    "precomputed": DistanceType.Precomputed
}


class DBSCAN(Base,
             InteropMixin,
             ClusterMixin,
             CMajorInputTagMixin):
    """
    DBSCAN is a very powerful yet fast clustering technique that finds clusters
    where data is concentrated. This allows DBSCAN to generalize to many
    problems if the datapoints tend to congregate in larger groups.

    cuML's DBSCAN expects an array-like object or cuDF DataFrame, and
    constructs an adjacency graph to compute the distances between close
    neighbours.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import DBSCAN
        >>> from cuml.cluster import DBSCAN
        >>>
        >>> import cudf
        >>> import numpy as np
        >>>
        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = np.asarray([1.0,2.0,5.0], dtype = np.float32)
        >>> gdf_float['1'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)
        >>> gdf_float['2'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)
        >>>
        >>> dbscan_float = DBSCAN(eps = 1.0, min_samples = 1)
        >>> dbscan_float.fit(gdf_float)
        DBSCAN()
        >>> dbscan_float.labels_
        0    0
        1    1
        2    2
        dtype: int32

    Parameters
    ----------
    eps : float (default = 0.5)
        The maximum distance between 2 points such they reside in the same
        neighborhood.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    min_samples : int (default = 5)
        The number of samples in a neighborhood such that this group can be
        considered as an important core point (including the point itself).
    metric: {'euclidean', 'cosine', 'precomputed'}, default = 'euclidean'
        The metric to use when calculating distances between points.
        If metric is 'precomputed', X is assumed to be a distance matrix
        and must be square.
        The input will be modified temporarily when cosine distance is used
        and the restored input matrix might not match completely
        due to numerical rounding.
    algorithm: {'brute', 'rbc'}, default = 'brute'
        The algorithm to be used by for nearest neighbor computations.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    max_mbytes_per_batch : (optional) int64
        Calculate batch size using no more than this number of megabytes for
        the pairwise distance computation. This enables the trade-off between
        runtime and memory usage for making the N^2 pairwise distance
        computations more tractable for large numbers of samples.
        If you are experiencing out of memory errors when running DBSCAN, you
        can set this value based on the memory size of your device.
        Note: this option does not set the maximum total memory used in the
        DBSCAN computation and so this value will not be able to be set to
        the total memory available on the device.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    calc_core_sample_indices : (optional) boolean (default = True)
        Indicates whether the indices of the core samples should be calculated.
        The the attribute `core_sample_indices_` will not be used, setting this
        to False will avoid unnecessary kernel launches

    Attributes
    ----------
    labels_ : array-like or cuDF series
        Which cluster each datapoint belongs to. Noisy samples are labeled as
        -1. Format depends on cuml global output type and estimator
        output_type.
    core_sample_indices_ : array-like or cuDF series
        The indices of the core samples. Only calculated if
        calc_core_sample_indices==True

    Notes
    -----
    DBSCAN is very sensitive to the distance metric it is used with, and a
    large assumption is that datapoints need to be concentrated in groups for
    clusters to be constructed.

    **Applications of DBSCAN**

        DBSCAN's main benefit is that the number of clusters is not a
        hyperparameter, and that it can find non-linearly shaped clusters.
        This also allows DBSCAN to be robust to noise.
        DBSCAN has been applied to analyzing particle collisions in the
        Large Hadron Collider, customer segmentation in marketing analyses,
        and much more.

    For additional docs, see `scikitlearn's DBSCAN
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_.
    """

    core_sample_indices_ = CumlArrayDescriptor(order="C")
    labels_ = CumlArrayDescriptor(order="C")

    _cpu_class_path = "sklearn.cluster.DBSCAN"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "eps",
            "min_samples",
            "max_mbytes_per_batch",
            "calc_core_sample_indices",
            "metric",
            "algorithm",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.metric):
            raise UnsupportedOnGPU
        elif model.metric not in _SUPPORTED_METRICS:
            raise UnsupportedOnGPU

        if model.algorithm not in ("auto", "brute"):
            raise UnsupportedOnGPU

        return {
            "eps": model.eps,
            "min_samples": model.min_samples,
            "metric": model.metric,
            "algorithm": "brute",
        }

    def _params_to_cpu(self):
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "algorithm": "brute",
        }

    def _attrs_from_cpu(self, model):
        return {
            "core_sample_indices_": to_gpu(model.core_sample_indices_, order="C"),
            "labels_": to_gpu(model.labels_, order="C"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "core_sample_indices_": to_cpu(self.core_sample_indices_, order="C"),
            "labels_": to_cpu(self.labels_, order="C"),
            **super()._attrs_to_cpu(model),
        }

    def __init__(self, *,
                 eps=0.5,
                 handle=None,
                 min_samples=5,
                 metric='euclidean',
                 algorithm='brute',
                 verbose=False,
                 max_mbytes_per_batch=None,
                 output_type=None,
                 calc_core_sample_indices=True):
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.eps = eps
        self.min_samples = min_samples
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.calc_core_sample_indices = calc_core_sample_indices
        self.metric = metric
        self.algorithm = algorithm

        # internal array attributes
        self.labels_ = None

        # One used when `self.calc_core_sample_indices == True`
        self.core_sample_indices_ = None

        # C++ API expects this to be numeric.
        if self.max_mbytes_per_batch is None:
            self.max_mbytes_per_batch = 0

    def _fit(self, X, out_dtype, opg, sample_weight,
             convert_dtype=True) -> "DBSCAN":
        """
        Protected auxiliary function for `fit`. Takes an additional parameter
        opg that is set to `False` for SG, `True` for OPG (multi-GPU)
        """
        if out_dtype not in ["int32", np.int32, "int64", np.int64]:
            raise ValueError(f"Invalid value for out_dtype: {out_dtype}. "
                             "Valid values are {'int32', 'int64', "
                             "np.int32, np.int64}")

        X_m, n_rows, self.n_features_in_, self.dtype = \
            input_to_cuml_array(
                X,
                order='C',
                convert_to_dtype=(np.float32 if convert_dtype
                                  else None),
                check_dtype=[np.float32, np.float64]
            )

        if n_rows == 0:
            raise ValueError("No rows in the input array. DBScan cannot be "
                             "fitted!")

        cdef uintptr_t input_ptr = X_m.ptr

        cdef uintptr_t sample_weight_ptr = <uintptr_t> NULL
        if sample_weight is not None:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(
                    sample_weight,
                    convert_to_dtype=(self.dtype if convert_dtype
                                      else None),
                    check_dtype=self.dtype,
                    check_rows=n_rows,
                    check_cols=1)
            sample_weight_ptr = sample_weight_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        self.labels_ = CumlArray.empty(n_rows, dtype=out_dtype,
                                       index=X_m.index)
        cdef uintptr_t labels_ptr = self.labels_.ptr

        cdef uintptr_t core_sample_indices_ptr = <uintptr_t> NULL

        # metric
        if (metric := _SUPPORTED_METRICS.get(self.metric.lower())) is None:
            raise ValueError(f"Invalid value for metric: {self.metric}")

        # algo
        algo_parsing = {
            "brute": EpsNnMethod.BRUTE_FORCE,
            "rbc": EpsNnMethod.RBC
        }
        if self.algorithm in algo_parsing:
            algorithm = algo_parsing[self.algorithm.lower()]
        else:
            raise ValueError("Invalid value for algorithm: {}"
                             .format(self.algorithm))

        # Create the output core_sample_indices only if needed
        if self.calc_core_sample_indices:
            self.core_sample_indices_ = \
                CumlArray.empty(n_rows, dtype=out_dtype)
            core_sample_indices_ptr = self.core_sample_indices_.ptr

        if self.dtype == np.float32:
            if out_dtype == "int32" or out_dtype is np.int32:
                fit(handle_[0],
                    <float*>input_ptr,
                    <int> n_rows,
                    <int> self.n_features_in_,
                    <float> self.eps,
                    <int> self.min_samples,
                    <DistanceType> metric,
                    <int*> labels_ptr,
                    <int*> core_sample_indices_ptr,
                    <float*> sample_weight_ptr,
                    <size_t>self.max_mbytes_per_batch,
                    <EpsNnMethod> algorithm,
                    <level_enum> self.verbose,
                    <bool> opg)
            else:
                fit(handle_[0],
                    <float*>input_ptr,
                    <int64_t> n_rows,
                    <int64_t> self.n_features_in_,
                    <float> self.eps,
                    <int> self.min_samples,
                    <DistanceType> metric,
                    <int64_t*> labels_ptr,
                    <int64_t*> core_sample_indices_ptr,
                    <float*> sample_weight_ptr,
                    <size_t>self.max_mbytes_per_batch,
                    <EpsNnMethod> algorithm,
                    <level_enum> self.verbose,
                    <bool> opg)

        else:
            if out_dtype == "int32" or out_dtype is np.int32:
                fit(handle_[0],
                    <double*>input_ptr,
                    <int> n_rows,
                    <int> self.n_features_in_,
                    <double> self.eps,
                    <int> self.min_samples,
                    <DistanceType> metric,
                    <int*> labels_ptr,
                    <int*> core_sample_indices_ptr,
                    <double*> sample_weight_ptr,
                    <size_t> self.max_mbytes_per_batch,
                    <EpsNnMethod> algorithm,
                    <level_enum> self.verbose,
                    <bool> opg)
            else:
                fit(handle_[0],
                    <double*>input_ptr,
                    <int64_t> n_rows,
                    <int64_t> self.n_features_in_,
                    <double> self.eps,
                    <int> self.min_samples,
                    <DistanceType> metric,
                    <int64_t*> labels_ptr,
                    <int64_t*> core_sample_indices_ptr,
                    <double*> sample_weight_ptr,
                    <size_t> self.max_mbytes_per_batch,
                    <EpsNnMethod> algorithm,
                    <level_enum> self.verbose,
                    <bool> opg)

        # make sure that the `fit` is complete before the following
        # delete call happens
        self.handle.sync()
        del X_m

        # Finally, resize the core_sample_indices array if necessary
        if self.calc_core_sample_indices:
            # Temp convert to cupy array (better than using `cupy.asarray`)
            with using_output_type("cupy"):
                # First get the min index. These have to monotonically
                # increasing, so the min index should be the first returned -1
                min_index = cp.argmin(self.core_sample_indices_).item()
                # Check for the case where there are no -1's
                if ((min_index == 0 and
                        self.core_sample_indices_[min_index].item() != -1)):
                    # Nothing to delete. The array has no -1's
                    pass
                else:
                    self.core_sample_indices_ = \
                        self.core_sample_indices_[:min_index]

        return self

    @generate_docstring(skip_parameters_heading=True)
    @warn_legacy_device_interop
    @deprecate_non_keyword_only("out_dtype", "convert_dtype")
    def fit(self, X, y=None, out_dtype="int32", sample_weight=None,
            convert_dtype=True) -> "DBSCAN":
        """
        Perform DBSCAN clustering from features.

        Parameters
        ----------
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}.

        sample_weight: array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at
            least min_samples is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            default: None (which is equivalent to weight 1 for all samples).
        """
        return self._fit(X, out_dtype, False, sample_weight)

    @generate_docstring(skip_parameters_heading=True,
                        return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster labels',
                                       'shape': '(n_samples, 1)'})
    @warn_legacy_device_interop
    @deprecate_non_keyword_only("out_dtype")
    def fit_predict(self, X, y=None, out_dtype="int32", sample_weight=None) -> CumlArray:
        """
        Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}.

        sample_weight: array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at
            least min_samples is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            default: None (which is equivalent to weight 1 for all samples).
        """
        self.fit(X, out_dtype=out_dtype, sample_weight=sample_weight)
        return self.labels_
