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

from sklearn.utils.fixes import signature
from collections import defaultdict

cdef extern from "spectral/spectral_c.h" namespace "ML":

    cdef void spectral_fit_embedding(float *input,
                   int n_rows,
                   int n_cols,
                   int n_neighbors,
                   int n_components,
                   float *embedding)


class SpectralEmbedding:
    """
    Create a DataFrame, fill it with data, and project it onto a low-dimensional
    space defined by the eigenvectors corresponding to the smallest n_components
    eigenvalues:

    .. code-block:: python

            import cudf
            from cuml import SpectralEmbedding
            import numpy as np

            gdf_float = cudf.DataFrame()
            gdf_float['0']=np.asarray([1.0,2.0,5.0],dtype=np.float32)
            gdf_float['1']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
            gdf_float['2']=np.asarray([4.0,2.0,1.0],dtype=np.float32)

            embed = SpectralEmbedding(eps=1.0, min_samples=1)
            embed.fit(gdf_float)
            print(embed.embedding_)

    Output:

    .. code-block:: python

            0    0
            1    1
            2    2

    For an additional example, see `the SpectralEmbedding notebook <https://github.com/rapidsai/cuml/blob/master/python/notebooks/spectral_embedding_demo.ipynb>`_.
    For additional docs, see `Scikitlearn's SpectralEmbedding <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html>`_.

    """

    def __init__(self, n_components=2, n_neighbors = 10, should_downcast = True):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_ = None
        self.embed_array_ = None
        self._should_downcast = should_downcast

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())


    def _downcast(self, X):

        if isinstance(X, cudf.DataFrame):
            dtype = np.dtype(X[X.columns[0]]._column.dtype)

            self.n_rows = len(X)
            self.n_cols = len(X._cols)

            if dtype != np.float32:
                if self._should_downcast:

                    new_cols = [(col,X._cols[col].astype(np.float32)) for col in X._cols]
                    overflowed = sum([len(colval[colval >= np.inf])  for colname, colval in new_cols])

                    if overflowed > 0:
                        raise Exception("Downcast to single-precision resulted in data loss.")

                    X = cudf.DataFrame(new_cols)

                else:
                    raise Exception("Input is double precision. Use 'should_downcast=True' "
                                    "if you'd like it to be automatically casted to single precision.")

            X = numba_utils.row_matrix(X)
        elif isinstance(X, np.ndarray):
            dtype = X.dtype
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

            if dtype != np.float32:
                if self._should_downcast:
                    X = X.astype(np.float32)
                    if len(X[X == np.inf]) > 0:
                        raise Exception("Downcast to single-precision resulted in data loss.")

                else:
                    raise Exception("Input is double precision. Use 'should_downcast=True' "
                                    "if you'd like it to be automatically casted to single precision.")

            X = cuda.to_device(X)
        else:
            raise Exception("Received unsupported input type " % type(X))

        return X


    def fit(self, X):
        """
            Perform DBSCAN clustering from features.

            Parameters
            ----------
            X : cuDF DataFrame or numpy ndarray
               Dense matrix (floats or doubles) of shape (n_samples, n_features)
        """

        if self.embedding_ is not None:
            del self.embedding_

        X_m = self._downcast(X)

        cdef uintptr_t input_ptr = self._get_ctype_ptr(X_m)

        self.embedding_array_ = cuda.to_device(
            np.zeros((self.n_rows,
                      self.n_components),
                      order = "C",
                      dtype=np.float32))

        cdef uintptr_t embedding_ptr = self._get_ctype_ptr(self.embedding_array_)

        spectral_fit_embedding(<float*>input_ptr,
                               <int> self.n_rows,
                               <int> self.n_cols,
                               <int> self.n_neighbors,
                               <int> self.n_components,
                               <float*> embedding_ptr)

        if isinstance(X, cudf.DataFrame):
            self.embedding_ = cudf.DataFrame()
            for i in range(0, self.embedding_array_.shape[1]):
                self.embedding_[str(i)] = self.embedding_array_[:,i]
        elif isinstance(X, np.ndarray):
            self.embedding_ = np.asarray(self.embedding_array_)

        del(X_m)
        return self

    def fit_transform(self, X):
        """
            Performs clustering on input_gdf and returns cluster labels.

            Parameters
            ----------
            X : cuDF DataFrame or numpy ndarray
              Dense matrix (floats or doubles) of shape (n_samples, n_features),

            Returns
            -------
            labels : cuDF Dataframe or numpy ndarray, shape (n_samples)
              low-dimensional embeddings
        """
        self.fit(X)
        return self.embedding_

    def get_params(self, deep=True):
        params = dict()
        variables = [ 'n_neighbors','n_components']
        for key in variables:
            var_value = getattr(self,key,None)
            params[key] = var_value
        return params



    def set_params(self, **params):
        if not params:
            return self
        current_params = {"n_neighbors": self.n_neighbors,"n_components":self.n_components}
        for key, value in params.items():
            if key not in current_params:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
                current_params[key] = value
        return self
