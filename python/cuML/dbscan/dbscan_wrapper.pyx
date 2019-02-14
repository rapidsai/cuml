#
# Copyright (c) 2018, NVIDIA CORPORATION.
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

cimport c_dbscan
import numpy as np
from numba import cuda
import cudf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t
from c_dbscan cimport *
# temporary import for numba_utils
from cuML import numba_utils


class DBSCAN:
    """
    Create a DataFrame, fill it with data, and compute DBSCAN:

    .. code-block:: python

            import cudf
            from cuml import DBSCAN
            import numpy as np

            gdf_float = cudf.DataFrame()
            gdf_float['0']=np.asarray([1.0,2.0,5.0],dtype=np.float32)
            gdf_float['1']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
            gdf_float['2']=np.asarray([4.0,2.0,1.0],dtype=np.float32)

            dbscan_float = DBSCAN(eps=1.0, min_samples=1)
            dbscan_float.fit(gdf_float)
            print(dbscan_float.labels_)

    Output:

    .. code-block:: python

            0    0
            1    1
            2    2

    For an additional example, see `the DBSCAN notebook <https://github.com/rapidsai/cuml/blob/master/python/notebooks/dbscan_demo.ipynb>`_. For additional docs, see `scikitlearn's DBSCAN <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_.

    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X):
        """
            Perform DBSCAN clustering from features or distance matrix.

            Parameters
            ----------
            X : cuDF DataFrame
               Dense matrix (floats or doubles) of shape (n_samples, n_features)
        """

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

        print("n_rows: "+ str(self.n_rows))

        input_ptr = self._get_ctype_ptr(X_m)

        self.labels_ = cudf.Series(np.zeros(self.n_rows, dtype=np.int32))
        self.labels_array = self.labels_._column._data.to_gpu_array()
        cdef uintptr_t labels_ptr = self._get_ctype_ptr(self.labels_array)
        print(hex(labels_ptr))
        if self.gdf_datatype.type == np.float32:
            c_dbscan.dbscanFit(<float*>input_ptr,
                               <int> self.n_rows,
                               <int> self.n_cols,
                               <float> self.eps,
                               <int> self.min_samples,
		               <int*> labels_ptr)
        else:
            c_dbscan.dbscanFit(<double*>input_ptr,
                               <int> self.n_rows,
                               <int> self.n_cols,
                               <double> self.eps,
                               <int> self.min_samples,
		               <int*> labels_ptr)
        #del(X_m)
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
