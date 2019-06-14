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

import cudf
import cuml
import ctypes
import numpy as np
import pandas as pd

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle

from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.memory cimport shared_ptr

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "tsne/tsne.h" namespace "ML":
    void TSNE(cumlHandle &handle,
              float *X,
              float *Y,
              int n,
              int p,
              int n_components,
              int n_neighbors,
              float perplexity,
              int perplexity_epochs,
              int perplexity_tol,
              float early_exaggeration,
              int exaggeration_iter,
              float min_gain,
              float eta,
              int epochs,
              float pre_momentum,
              float post_momentum,
              long long seed,
              bool initialize_embeddings) except +


class TSNE_py:

    def __cinit__(self,
                  n_neighbors=90,
                  n_components=2,
                  epochs=150,
                  perplexity=30.0,
                  perplexity_epochs=100,
                  perplexity_tol=1e-5,
                  early_exaggeration=12.0,
                  exaggeration_iter=250,
                  min_gain=0.01,
                  eta=500.0,
                  pre_momentum=0.8,
                  post_momentum=0.5,
                  seed=-1,
                  verbose=False,
                  should_downcast=True,
                  handle=None):

        self.handle = handle

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.epochs = epochs
        self.perplexity = perplexity
        self.perplexity_epochs = perplexity_epochs
        self.perplexity_tol = perplexity_tol
        self.early_exaggeration = early_exaggeration
        self.exaggeration_iter = exaggeration_iter
        self.min_gain = min_gain
        self.eta = eta
        self.pre_momentum = pre_momentum
        self.post_momentum = post_momentum
        self.seed = seed
        self.verbose = verbose

        self._should_downcast = should_downcast

    def fit(self, X):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            X contains a sample per row.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : array-like (device or host) shape = (n_samples, 1)
            y contains a label per row.
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        """

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        if self._should_downcast:
            X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', convert_to_dtype=np.float32)
        else:
            X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', check_dtype=np.float32)

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        self.n_neighbors = min(n_rows, self.n_neighbors)
        self.n_dims = n_cols
        self.raw_data = X_ctype

        self.arr_embed = cuda.to_device(zeros((X_m.shape[0],
                                        self.n_components),
                                        order="C", dtype=np.float32))
        self.embeddings = self.arr_embed.device_ctypes_pointer.value

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t X_ptr = self.raw_data
        cdef uintptr_t embed_ptr = self.embeddings

        cdef uintptr_t y_raw
        TSNE(handle_[0],
             <float*> X_ptr,
             <float*>embed_ptr,
             <int> X_m.shape[0],
             <int> X_m.shape[1],
             <int>self.n_components,
             <int>self.n_neighbors,
             <float>self.perplexity,
             <int> self.perplexity_epochs,
             <int> self.perplexity_tol,
             <float>self.early_exaggeration,
             <int> self.exaggeration_iter,
             <float> self.min_gain,
             <float> self.eta,
             <int> self.epochs,
             <float> self.pre_momentum,
             <float> self.post_momentum,
             <long long> self.seed,
             <bool>True)

        del X_m

        return self

    def fit_transform(self, X):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            X contains a sample per row.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X)

        if isinstance(X, cudf.DataFrame):
            ret = cudf.DataFrame()
            for i in range(0, self.arr_embed.shape[1]):
                ret[str(i)] = self.arr_embed[:, i]
        elif isinstance(X, np.ndarray):
            ret = np.asarray(self.arr_embed)

        return ret
