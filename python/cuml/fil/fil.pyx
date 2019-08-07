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

import copy
import cudf
import ctypes
import math
import numpy as np
import warnings

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros
cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle

cdef extern from "fil/fil.h" namespace "ML::fil":

    cdef enum algo_t:
        NAIVE,
        TREE_REORG,
        BATCH_TREE_REORG

    cdef enum output_t:
        RAW,
        PROB,
        CLASS

    cdef struct dense_node_t:
        float val
        int bits

    cdef struct forest:
        pass

    ctypedef forest* forest_t

    cdef struct forest_params_t:
        dense_node_t* nodes
        int depth
        int ntrees
        int cols
        algo_t algo
        output_t output
        float threshold
        pass

    cdef struct treelite_params_t:
        algo_t algo
        bool output_class
        float threshold

    cdef void free(cumlHandle& handle,
                   forest_t)

    cdef void predict(cumlHandle& handle,
                      forest_t,
                      float*,
                      float*,
                      size_t)

    cdef forest_t from_treelite(cumlHandle& handle,
                                forest_t*,
                                ModelHandle,
                                treelite_params_t*)

cdef class FIL_impl():

    cpdef object handle
    cdef forest_t* forest_pointer
    cdef forest_t forest_data
    cdef object algo
    cdef object threshold

    def __cinit__(self,
                  algo=0,
                  threshold=0.0,
                  handle=None):

        self.algo = algo
        self.threshold = threshold
        self.handle = handle
        self.forest_pointer = NULL

    def predict(self, X,):
        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, _, X_dtype = \
            input_to_dev_array(X, order='C')

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        preds = cudf.Series(zeros(n_rows, dtype=np.float32))
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)
        predict(handle_[0],
                self.forest_data,
                <float*> preds_ptr,
                <float*> X_ptr,
                <size_t> n_rows)
        self.handle.sync()
        # synchronous w/o a stream
        return preds

    def __cdel__(self):

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        free(handle_[0],
             self.forest_data)

        return self

    def from_treelite(self, model, output_class):

        cdef treelite_params_t treelite_params
        treelite_params.output_class = output_class
        treelite_params.threshold = self.threshold
        treelite_params.algo = self.algo
        self.forest_pointer = <forest_t*><size_t> model.handle.value
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = model.handle.value
        self.forest_data = from_treelite(handle_[0],
                                         self.forest_pointer,
                                         <ModelHandle> model_ptr,
                                         &treelite_params)

        return self

class FIL(Base):


    def __init__(self,
                 algo=0,
                 threshold=0.0,
                 handle=None):

        super(FIL, self).__init__(handle)
        self.algo_type = algo
        self.threshold = threshold

        self._impl = FIL_impl(algo,
                              threshold,
                              self.handle)

    def predict(self, X):

        return self._impl.predict(X)

    def __del__(self):

        return self._impl.__cdel__()

    def from_treelite(self, model, output_class):
        return self._impl.from_treelite(model, output_class)
