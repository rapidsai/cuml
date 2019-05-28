#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array

from collections import defaultdict

cdef extern from "dbscan/dbscan.hpp" namespace "ML":

    cdef void dbscanFit(cumlHandle& handle,
                        float *input,
                        int n_rows,
                        int n_cols,
                        float eps,
                        int min_pts,
                        int *labels,
                        size_t max_bytes_per_batch,
                        bool verbose)

    cdef void dbscanFit(cumlHandle& handle,
                        double *input,
                        int n_rows,
                        int n_cols,
                        double eps,
                        int min_pts,
                        int *labels,
                        size_t max_bytes_per_batch,
                        bool verbose)


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
    max_bytes_per_batch : (optional) int64
        Calculate batch size using no more than this number of bytes for the
        pairwise distance computation. This enables the trade-off between
        runtime and memory usage for making the N^2 pairwise distance
        computations more tractable for large numbers of samples.
        If you are experiencing out of memory errors when running DBSCAN, you
        can set this value based on the memory size of your device.
        Note: this option does not set the maximum total memory used in the
        DBSCAN computation and so this value will not
        be able to be set to the total memory available on the device.

    Attributes
    -----------
    labels_ : array
        Which cluster each datapoint belongs to. Noisy samples are labeled as
        -1.

    Notes
    ------
    DBSCAN is very sensitive to the distance metric it is used with, and a
    large assumption is that datapoints need to be concentrated in groups for
    clusters to be constructed.

    **Applications of DBSCAN**

        DBSCAN's main benefit is that the number of clusters is not a
        hyperparameter, and that it can find non-linearly shaped clusters.
        This also allows DBSCAN to be robust to noise.
        DBSCAN has been applied to analyzing particle collisons in the
        Large Hadron Collider, customer segmentation in marketing analyses,
        and much more.


    For an additional example, see `the DBSCAN notebook
    <https://github.com/rapidsai/notebooks/blob/master/cuml/dbscan_demo.ipynb>`_.
    For additional docs, see `scikitlearn's DBSCAN
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_.
    """

    def __init__(self, eps=0.5, handle=None, min_samples=5, verbose=False,
                 max_bytes_per_batch=None):
        super(DBSCAN, self).__init__(handle, verbose)
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.max_bytes_per_batch = max_bytes_per_batch
        self.verbose = verbose

        # C++ API expects this to be numeric.
        if self.max_bytes_per_batch is None:
            self.max_bytes_per_batch = 0

    def __getattr__(self, attr):
        if attr == 'labels_array':
            return self.labels_._column._data.mem

    def fit(self, X):
        """
        Perform DBSCAN clustering from features.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats or doubles) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
        """

        if self.labels_ is not None:
            del self.labels_

        cdef uintptr_t input_ptr

        X_m, input_ptr, n_rows, n_cols, self.dtype = \
            input_to_dev_array(X, order='C')

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        self.labels_ = cudf.Series(np.zeros(n_rows, dtype=np.int32))
        cdef uintptr_t labels_ptr = get_cudf_column_ptr(self.labels_)

        if self.dtype.type == np.float32:
            dbscanFit(handle_[0],
                      <float*>input_ptr,
                      <int> n_rows,
                      <int> n_cols,
                      <float> self.eps,
                      <int> self.min_samples,
                      <int*> labels_ptr,
                      <size_t>self.max_bytes_per_batch,
                      <bool>self.verbose)
        else:
            dbscanFit(handle_[0],
                      <double*>input_ptr,
                      <int> n_rows,
                      <int> n_cols,
                      <double> self.eps,
                      <int> self.min_samples,
                      <int*> labels_ptr,
                      <size_t> self.max_bytes_per_batch,
                      <bool>self.verbose)
        # make sure that the `dbscanFit` is complete before the following
        # delete call happens
        self.handle.sync()
        del(X_m)
        return self

    def fit_predict(self, X):
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
        self.fit(X)
        return self.labels_

    def get_param_names(self):
        return ["eps", "min_samples"]
