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

from numba import cuda
from cuml import numba_utils

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from collections import defaultdict

cdef extern from "dbscan/dbscan_c.h" namespace "ML":

    cdef void dbscanFit(float *input,
                   int n_rows,
                   int n_cols,
                   float eps,
                   int min_pts,
                   int *labels)

    cdef void dbscanFit(double *input,
                   int n_rows,
                   int n_cols,
                   double eps,
                   int min_pts,
                   int *labels)


class DBSCAN:
	"""
	DBSCAN is a very powerful yet fast clustering technique that finds clusters where
	data is concentrated. This allows DBSCAN to generalize to many problems if the
	datapoints tend to congregate in larger groups.

	cuML's DBSCAN expects a cuDF DataFrame, and constructs an adjacency graph to compute
	the distances between close neighbours.
	
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
		The maximum distance between 2 points such they reside in the same neighborhood.
	min_samples : int (default = 5)
		The number of samples in a neighborhood such that this group can be considered as
		an important core point (including the point itself).

	Attributes
	-----------
	labels_ : array
		Which cluster each datapoint belongs to. Noisy samples are labeled as -1.

	Notes
	------
	DBSCAN is very sensitive to the distance metric it is used with, and a large assumption
	is that datapoints need to be concentrated in groups for clusters to be constructed.
	
	**Applications of DBSCAN**

		DBSCAN's main benefit is that the number of clusters is not a hyperparameter, and that
		it can find non-linearly shaped clusters. This also allows DBSCAN to be robust to noise.
		DBSCAN has been applied to analyzing particle collisons in the Large Hadron Collider,
		customer segmentation in marketing analyses, and much more.


	For an additional example, see `the DBSCAN notebook <https://github.com/rapidsai/notebooks/blob/master/cuml/dbscan_demo.ipynb>`_.
	For additional docs, see `scikitlearn's DBSCAN <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_.
	"""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.labels_array = None

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X):
        """
            Perform DBSCAN clustering from features.

            Parameters
            ----------
            X : cuDF DataFrame
               Dense matrix (floats or doubles) of shape (n_samples, n_features)
        """

        if self.labels_ is not None:
            del self.labels_

        if self.labels_array is not None:
            del self.labels_array

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = numba_utils.row_matrix(X)
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]


        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        self.labels_ = cudf.Series(np.zeros(self.n_rows, dtype=np.int32))
        self.labels_array = self.labels_._column._data.to_gpu_array()
        cdef uintptr_t labels_ptr = self._get_ctype_ptr(self.labels_array)
        if self.gdf_datatype.type == np.float32:
            dbscanFit(<float*>input_ptr,
                               <int> self.n_rows,
                               <int> self.n_cols,
                               <float> self.eps,
                               <int> self.min_samples,
		               <int*> labels_ptr)
        else:
            dbscanFit(<double*>input_ptr,
                               <int> self.n_rows,
                               <int> self.n_cols,
                               <double> self.eps,
                               <int> self.min_samples,
		               <int*> labels_ptr)
        del(X_m)
        return self

    def fit_predict(self, X):
        """
            Performs clustering on input_gdf and returns cluster labels.

            Parameters
            ----------
            X : cuDF DataFrame
              Dense matrix (floats or doubles) of shape (n_samples, n_features),

            Returns
            -------
            y : cuDF Series, shape (n_samples)
              cluster labels
        """
        self.fit(X)
        return self.labels_

    def get_params(self, deep=True):
	"""
	Sklearn style return parameter state

	Parameters
	-----------
	deep : boolean (default = True)
	"""
        params = dict()
        variables = [ 'eps','min_samples']
        for key in variables:
            var_value = getattr(self,key,None)
            params[key] = var_value
        return params



    def set_params(self, **params):
	"""
	Sklearn style set parameter state to dictionary of params.

	Parameters
	-----------
	params : dict of new params
	"""
        if not params:
            return self
        current_params = {"eps": self.eps,"min_samples":self.min_samples}
        for key, value in params.items():
            if key not in current_params:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
                current_params[key] = value
        return self
