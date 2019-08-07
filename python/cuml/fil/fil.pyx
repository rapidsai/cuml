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

    def predict(self, X, preds=None):
        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, _, X_dtype = \
            input_to_dev_array(X, order='C')

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if preds is None:
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

    def from_treelite(self, model, output_class):
        cdef treelite_params_t treelite_params
        treelite_params.output_class = output_class
        treelite_params.threshold = self.threshold
        treelite_params.algo = self.algo
        cdef forest_t* forest_pointer =\
            <forest_t*><size_t> model.handle.value
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = model.handle.value
        self.forest_data = from_treelite(handle_[0],
                                         forest_pointer,
                                         <ModelHandle> model_ptr,
                                         &treelite_params)
        return self

    def __cdel__(self):
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        free(handle_[0],
             self.forest_data)
        return self


class FIL(Base):
    """
    Parameters
    ----------
    model : the model information in the treelite format
       loaded from a saved model using the treelite API
       https://treelite.readthedocs.io/en/latest/treelite-api.html
    output_class: boolean
       True or False
    algo : 0 = NAIVE, 1 = TREE_REORG, 2 = BATCH_TREE_REORG
    threshold : threshold is used to for classification
       if output == OUTPUT_CLASS, else it is ignored
    """
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

    def predict(self, X, preds=None):
        """
        Predicts the labels for X on the model that was loaded
        using from_treelite function.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats or doubles) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
        preds: NumPy or cuDF dataframe
           Dense vector (int) of shape (n_samples, 1)
        Returns
        ----------
        labels predicted by the model
        """
        return self._impl.predict(X, preds)

    def from_treelite(self, model, output_class):
        """
        Creates a FIL model using the treelite model
        passed to the function.
        Parameters
        ----------
        model : the model information in the treelite format
           loaded from a saved model using the treelite API
           https://treelite.readthedocs.io/en/latest/treelite-api.html
        output_class: boolean
           True or False
        """
        return self._impl.from_treelite(model, output_class)


def from_treelite_direct(model, algo=0, output_class=True,
                         threshold=0.5, handle=None):
    """
    Creates a FIL model using the treelite model
    passed to the function.
    Note : from cuml import fil
           forest = fil.from_treelite_direct(params)
           # in order to predict
           predicted_labels = forest.predict(X_test)

    Note : do not reuse or overwrite a tl_model variable which is
       used to load a saved treelite model.
       This will cause Segmentation errors.

    Parameters
    ----------
    model : the model information in the treelite format
       loaded from a saved model using the treelite API
       https://treelite.readthedocs.io/en/latest/treelite-api.html
    output_class: boolean
       True or False
    algo : 0 = NAIVE, 1 = TREE_REORG, 2 = BATCH_TREE_REORG
    threshold : threshold is used to for classification if
       output == OUTPUT_CLASS, else it is ignored
    handle : cuml.Handle
       If it is None, a new one is created just for this class.
    """
    fil_model = FIL(algo=algo, threshold=threshold, handle=handle)
    fil_model.from_treelite(model=model, output_class=output_class)
    return fil_model
