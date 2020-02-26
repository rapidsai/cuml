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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import cudf
import numpy as np

from libcpp cimport bool
from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_cuml_array

from collections import defaultdict

cdef extern from "cuml/cluster/dbscan.hpp" namespace "ML":

    cdef void dbscanFit(cumlHandle& handle,
                        float *input,
                        int n_rows,
                        int n_cols,
                        float eps,
                        int min_pts,
                        int *labels,
                        size_t max_mbytes_per_batch,
                        bool verbose) except +

    cdef void dbscanFit(cumlHandle& handle,
                        double *input,
                        int n_rows,
                        int n_cols,
                        double eps,
                        int min_pts,
                        int *labels,
                        size_t max_mbytes_per_batch,
                        bool verbose) except +

    cdef void dbscanFit(cumlHandle& handle,
                        float *input,
                        int64_t n_rows,
                        int64_t n_cols,
                        double eps,
                        int min_pts,
                        int64_t *labels,
                        size_t max_mbytes_per_batch,
                        bool verbose) except +

    cdef void dbscanFit(cumlHandle& handle,
                        double *input,
                        int64_t n_rows,
                        int64_t n_cols,
                        double eps,
                        int min_pts,
                        int64_t *labels,
                        size_t max_mbytes_per_batch,
                        bool verbose) except +


class DBSCAN(Base):
    """
    DBSCAN is a very powerful yet fast clustering technique that finds clusters
    where data is concentrated. This allows DBSCAN to generalize to many
    problems if the datapoints tend to congregate in larger groups.

    cuML's DBSCAN expects an array-like object or cuDF DataFrame, and
    constructs an adjacency graph to compute the distances between close
    neighbours.

    Examples
    ---------

    .. code-block:: python

            # Both import methods supported
            from cuml import DBSCAN
            from cuml.cluster import DBSCAN

            import cudf
            import numpy as np

            gdf_float = cudf.DataFrame()
            gdf_float['0'] = np.asarray([1.0,2.0,5.0], dtype = np.float32)
            gdf_float['1'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)
            gdf_float['2'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)

            dbscan_float = DBSCAN(eps = 1.0, min_samples = 1)
            dbscan_float.fit(gdf_float)
            print(dbscan_float.labels_)

    Output:

    .. code-block:: python

            0    0
            1    1
            2    2

    Parameters
    -----------
    eps : float (default = 0.5)
        The maximum distance between 2 points such they reside in the same
        neighborhood.
    handle : cuml.Handle
        If it is None, a new one is created just for this class
    min_samples : int (default = 5)
        The number of samples in a neighborhood such that this group can be
        considered as an important core point (including the point itself).
    verbose : bool
        Whether to print debug spews
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
    output_type : (optional) {'input', 'cudf', 'cupy', 'numpy'} default = None
        Use it to control output type of the results and attributes.
        If None it'll inherit the output type set at the
        module level, cuml.output_type. If that has not been changed, by
        default the estimator will mirror the type of the data used for each
        fit or predict call.
        If set, the estimator will override the global option for its behavior.

    Attributes
    -----------
    labels_ : array-like or cuDF series
        Which cluster each datapoint belongs to. Noisy samples are labeled as
        -1. Format depends on cuml global output type and estimator
        output_type.

    Notes
    ------
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


    For an additional example, see `the DBSCAN notebook
    <https://github.com/rapidsai/notebooks/blob/master/cuml/dbscan_demo.ipynb>`_.
    For additional docs, see `scikitlearn's DBSCAN
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_.
    """

    def __init__(self, eps=0.5, handle=None, min_samples=5, verbose=False,
                 max_mbytes_per_batch=None, output_type=None):
        super(DBSCAN, self).__init__(handle, verbose, output_type)
        self.eps = eps
        self.min_samples = min_samples
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.verbose = verbose

        # internal array attributes
        self._labels_ = None  # accessed via estimator.labels_

        # C++ API expects this to be numeric.
        if self.max_mbytes_per_batch is None:
            self.max_mbytes_per_batch = 0

    def fit(self, X, out_dtype="int32"):
        """
        Perform DBSCAN clustering from features.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats or doubles) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}. When the number of samples exceed
        """

        self._set_output_type(X)

        if self._labels_ is not None:
            del self._labels_

        if out_dtype not in ["int32", np.int32, "int64", np.int64]:
            raise ValueError("Invalid value for out_dtype. "
                             "Valid values are {'int32', 'int64', "
                             "np.int32, np.int64}")

        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='C',
                                check_dtype=[np.float32, np.float64])

        cdef uintptr_t input_ptr = X_m.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        self._labels_ = CumlArray.empty(n_rows, dtype=out_dtype)
        cdef uintptr_t labels_ptr = self._labels_.ptr

        if self.dtype == np.float32:
            if out_dtype is "int32" or out_dtype is np.int32:
                dbscanFit(handle_[0],
                          <float*>input_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <float> self.eps,
                          <int> self.min_samples,
                          <int*> labels_ptr,
                          <size_t>self.max_mbytes_per_batch,
                          <bool>self.verbose)
            else:
                dbscanFit(handle_[0],
                          <float*>input_ptr,
                          <int64_t> n_rows,
                          <int64_t> n_cols,
                          <float> self.eps,
                          <int> self.min_samples,
                          <int64_t*> labels_ptr,
                          <size_t>self.max_mbytes_per_batch,
                          <bool>self.verbose)

        else:
            if out_dtype is "int32" or out_dtype is np.int32:
                dbscanFit(handle_[0],
                          <double*>input_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <double> self.eps,
                          <int> self.min_samples,
                          <int*> labels_ptr,
                          <size_t> self.max_mbytes_per_batch,
                          <bool>self.verbose)
            else:
                dbscanFit(handle_[0],
                          <double*>input_ptr,
                          <int64_t> n_rows,
                          <int64_t> n_cols,
                          <double> self.eps,
                          <int> self.min_samples,
                          <int64_t*> labels_ptr,
                          <size_t> self.max_mbytes_per_batch,
                          <bool>self.verbose)

        # make sure that the `dbscanFit` is complete before the following
        # delete call happens
        self.handle.sync()
        del(X_m)
        return self

    def fit_predict(self, X, out_dtype="int32"):
        """
        Performs clustering on input_gdf and returns cluster labels.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
          Dense matrix (floats or doubles) of shape (n_samples, n_features)
          Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
          ndarray, cuda array interface compliant array like CuPy

        Returns
        -------
        y : cuDF Series, shape (n_samples)
          cluster labels
        """
        self.fit(X, out_dtype)
        return self.labels_

    def get_param_names(self):
        return ["eps", "min_samples"]
