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

    cdef void dense_node_init(dense_node_t*,
                              float,
                              float,
                              int,
                              bool,
                              bool)

    cdef void dense_node_decode(dense_node_t*,
                                float*,
                                float*,
                                int*,
                                bool*,
                                bool*)

    cdef void init_dense(cumlHandle& handle,
                         forest_t*,
                         forest_params_t*)

    cdef void free(cumlHandle& handle,
                   forest_t)

    cdef void predict(cumlHandle& handle,
                      forest_t,
                      float*,
                      float*,
                      size_t)


cdef class FIL_impl():

    cpdef object handle
    cdef dense_node_t* node_info
    cdef forest_t* forest_pointer
    cdef forest_t forest_data
    cdef forest_params_t* params
    cdef object nan_prob
    cdef object depth
    cdef object num_trees
    cdef object leaf_prob
    cdef object algo
    cdef object output
    cdef object threshold
    cdef object seed
    cdef object tolerance
    cdef object new_forest
    cdef object n_cols
    cdef object dtype
    cdef object weights
    cdef object fid
    cdef object def_left
    cdef object is_leaf

    def __cinit__(self, nan_prob=0.05, depth=8, n_estimators=50,
                  leaf_prob=0.05, output=0, algo=0,
                  threshold=0.0, seed=42, tolerance=2e-3,
                  new_forest=None, handle=None, verbose=False):

        self.nan_prob = nan_prob
        self.depth = depth
        self.num_trees = n_estimators
        self.leaf_prob = leaf_prob
        self.output = output
        self.algo = algo
        self.threshold = threshold
        self.seed = seed
        self.tolerance = tolerance
        self.new_forest = new_forest
        self.handle = handle
        self.verbose = verbose
        self.forest_pointer = NULL
        self.params = NULL
        self.node_info = NULL

    def dense_node_init(self, tree_node_info, weights,
                        fid, def_left, is_leaf):

            self.node_info = tree_node_info
            self.weights = <float*> weights
            self.fid = fid
            self.def_left = def_left
            self.is_leaf = is_leaf

            dense_node_init(<dense_node_t*> self.node_info,
                            <float*> self.weights,
                            <float> self.threshold,
                            <int> self.fid,
                            <bool> self.def_left,
                            <bool> self.is_leaf)
            return self

    def dense_node_decode(self):

        dense_node_decode(<dense_node_t*> self.node_info,
                          <float*> self.weights,
                          <float*> self.threshold,
                          <int*> self.fid,
                          <bool*> self.def_left,
                          s<bool*> self.is_leaf)
        return self

    def init_dense(self, X):

        cdef uintptr_t X_ptr  # , y_ptr
        # y_m, y_ptr, _, _, y_dtype = input_to_dev_array(y)

        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        self.params.depth = self.depth
        self.params.ntrees = self.num_trees
        self.params.cols = self.n_cols
        self.params.algo = self.algo
        self.params.output = self.output
        self.params.threshold = self.threshold

        init_dense(handle_[0],
                   self.forest_pointer,
                   self.params)

        return self

    def predict(self, X):
        cdef uintptr_t X_ptr  # , y_ptr
        # y_m, y_ptr, _, _, y_dtype = input_to_dev_array(y)

        X_m, X_ptr, n_rows, _, _ = \
            input_to_dev_array(X, order='F')

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        preds = np.zeros(n_rows, dtype=np.int32)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)

        predict(handle_[0],
                self.forest_data,
                <float*> preds_ptr,
                <float*> X_ptr,
                <size_t> n_rows)

        return preds

    def free(self):

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        free(handle_[0],
             self.forest_data)

        return self


class FIL(Base):
        # n_estimators = num_trees
        # output_type = output_t
        # algo_type = algo_t
    def __init__(self, nan_prob=0.05, depth=8, n_estimators=50,
                 leaf_prob=0.05, output_type=0, algo_type=0,
                 threshold=0.0, seed=42, tolerance=2e-3,
                 handle=None, verbose=False):

        super(FIL, self).__init__(handle, verbose)
        self.nan_prob = nan_prob
        self.depth = depth
        self.num_trees = n_estimators
        self.leaf_prob = leaf_prob
        self.output_type = output_type
        self.algo_type = algo_type
        self.threshold = threshold
        self.seed = seed
        self.tolerance = tolerance

        self._impl = FIL_impl(nan_prob, depth, n_estimators,
                              leaf_prob, output_type, algo_type,
                              threshold, seed, tolerance,
                              self.handle)

    def dense_node_init(self, tree_node_info, weights,
                        fid, def_left, is_leaf):

        return self._impl.dense_node_init(self, tree_node_info, weights,
                                          fid, def_left, is_leaf)

    def dense_node_decode(self):

        return self._impl.dense_node_decode()

    def init_dense(self, X):

        return self._impl.init_dense(X)

    def predict(self, X):

        return self._impl.predict(X)

    def free(self):

        return self._impl.free()
