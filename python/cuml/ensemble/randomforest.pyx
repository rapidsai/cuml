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
import cupy
import numpy as np

from numba import cuda

from cuml.utils import get_cudf_column_ptr,\
     get_dev_array_ptr, input_to_dev_array,\
     zeros

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
cimport cuml.common.handle
cimport cuml.common.cuda


cdef extern from "randomforest/randomforest.h" namespace "ML":

    cdef struct RF_metrics:
        float accuracy

    cdef enum RF_type:
        CLASSIFICATION,
        REGRESSION

    cdef struct RF_params:
        pass

    cdef cppclass rf:
        pass

    cdef cppclass rfClassifier[T]:
        rfClassifier(RF_params) except +

    cdef void fit(cumlHandle & handle,
                  rfClassifier[float] *,
                  float *,
                  int,
                  int,
                  int *,
                  int) except +

    cdef void fit(cumlHandle& handle,
                  rfClassifier[double] *,
                  double *,
                  int,
                  int,
                  int *,
                  int) except +

    cdef void predict(cumlHandle& handle,
                      rfClassifier[float] *,
                      float *,
                      int,
                      int,
                      int *,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      rfClassifier[double] *,
                      double *,
                      int,
                      int,
                      int *,
                      bool) except +

    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   rfClassifier[float] *, float *, int *,
                                   int, int, int *, bool)
    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   rfClassifier[double] *, double *, int *,
                                   int, int, int *, bool)

    cdef RF_params set_rf_class_obj(int, int, float,
                                    int, int, int,
                                    bool, bool, int, int) except +


cdef class RandomForest_impl():
    """
    Description and example code
    split_algo = 0 for HIST, 1 for GLOBAL_QUANTILE and 3 for SPLIT_ALGO_END
    """
    cpdef object handle
    cdef rfClassifier[float] *rf_classifier32
    cdef rfClassifier[double] *rf_classifier64
    cdef uintptr_t preds_ptr
    cdef object n_estimators
    cdef object max_depth
    cdef object max_features
    cdef object min_samples_split
    cdef object n_bins
    cdef object split_algo
    cdef object min_rows_per_node
    cdef object bootstrap
    cdef object bootstrap_features
    cdef object type_model
    cdef object verbose
    cdef object n_cols
    cdef object rows_sample
    cdef object max_leaves
    cdef object gdf_datatype
    cdef object stats
    cdef object dtype

    def __cinit__(self, n_estimators=10, max_depth=-1, handle=None,
                  max_features=1.0, min_samples_split=2, n_bins=8,
                  split_algo=0, min_rows_per_node=2,
                  bootstrap=True, bootstrap_features=False,
                  type_model="classifier", verbose=False,
                  rows_sample=1.0, max_leaves=-1,
                  gdf_datatype=None):

        self.handle = handle
        self.split_algo = split_algo
        self.min_rows_per_node = min_rows_per_node
        self.bootstrap_features = bootstrap_features
        self.rows_sample = rows_sample
        self.max_leaves = max_leaves
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.type_model = self._get_type(type_model)
        self.bootstrap = bootstrap
        self.verbose = False
        self.n_bins = n_bins
        self.rf_classifier32 = NULL
        self.rf_classifier64 = NULL
        self.n_cols = None

    def _get_type(self, type_model):
        if type_model == "classifier":
            self.type_model = 1
        else:
            self.type_model = 0

    """
    TBD:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
    """

    def fit(self, X, y):

        cdef uintptr_t X_ptr, y_ptr

        if self.rf_classifier32 != NULL:
            del self.rf_classifier32

        if self.rf_classifier64 != NULL:
            del self.rf_classifier64

        if y.dtype != np.int32:
            raise TypeError(" The labels need to have dtype = np.int32")

        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')
        y_m, y_ptr, _, _, _ = input_to_dev_array(y)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        unique_labels = cupy.unique(y)
        num_unique_labels = (unique_labels).__len__()
        for i in range(num_unique_labels):
            if i not in unique_labels:
                raise ValueError(" The labels need "
                                 "to be from 0 to num_unique_label values")

        rf_param = set_rf_class_obj(<int> self.max_depth,
                                    <int> self.max_leaves,
                                    <float> self.max_features,
                                    <int> self.n_bins,
                                    <int> self.split_algo,
                                    <int> self.min_rows_per_node,
                                    <bool> self.bootstrap_features,
                                    <bool> self.bootstrap,
                                    <int> self.n_estimators,
                                    <int> self.rows_sample)

        self.rf_classifier32 = new \
            rfClassifier[float](rf_param)
        self.rf_classifier64 = new \
            rfClassifier[double](rf_param)

        if self.dtype == np.float32:
            fit(handle_[0],
                self.rf_classifier32,
                <float*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> num_unique_labels)
        else:
            fit(handle_[0],
                self.rf_classifier64,
                <double*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> num_unique_labels)

        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self

    def predict(self, X):

        cdef uintptr_t X_ptr
        X_ptr = X.ctypes.data
        n_rows, n_cols = np.shape(X)
        if n_cols != self.n_cols:
            raise ValueError(" The number of columns/features in the training"
                             " and test data should be the same ")

        preds = np.zeros(n_rows,
                         dtype=np.int32)

        cdef uintptr_t preds_ptr = preds.ctypes.data
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            predict(handle_[0],
                    self.rf_classifier32,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    self.rf_classifier64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)

        else:
            raise TypeError("supports only float32 and float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        return preds

    def cross_validate(self, X, y):

        cdef uintptr_t X_ptr, y_ptr
        X_ptr = X.ctypes.data
        y_ptr = y.ctypes.data
        n_rows, n_cols = np.shape(X)

        if n_cols != self.n_cols:
            raise ValueError(" The number of columns/features in the training"
                             " and test data should be the same ")

        preds = np.zeros(n_rows,
                         dtype=np.int32)

        cdef uintptr_t preds_ptr = (preds).ctypes.data

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            self.stats = cross_validate(handle_[0],
                                        self.rf_classifier32,
                                        <float*> X_ptr,
                                        <int*> y_ptr,
                                        <int> n_rows,
                                        <int> n_cols,
                                        <int*> preds_ptr,
                                        <bool> self.verbose)

        elif self.dtype == np.float64:
            self.stats = cross_validate(handle_[0],
                                        self.rf_classifier64,
                                        <double*> X_ptr,
                                        <int*> y_ptr,
                                        <int> n_rows,
                                        <int> n_cols,
                                        <int*> preds_ptr,
                                        <bool> self.verbose)

        self.handle.sync()
        return self.stats


class RandomForestClassifier(Base):

    """
    Implements a Random Forest classifier model
    which fits multiple decision tree classifiers.
    The user is responsible for setting the various
    state variables to appropriate values.
    The model at the moment uses only numpy arrays as inputs.


    Examples
    ---------
    .. code-block:: python

            import numpy as np
            from cuml.test.utils import get_handle
            from cuml.ensemble import RandomForestClassifier as curfc
            from cuml.test.utils import get_handle

            X = np.asarray([[1,2],[10,20],[4,8],[50,70]], dtype=np.float32)
            y = np.asarray([0,1,0,1], dtype=np.int32)
            handle, stream = get_handle(True)

            cuml_model = curfc(max_features=1.0,
                               n_bins=2, split_algo=0, min_rows_per_node=2,
                               n_estimators=40, handle=handle)
            cuml_model.fit(X,y)
            cuml_predict = cuml_model.predict(X)

            print("Predicted labels : ", cuml_predict)

    Output:
    .. code-block:: python

            Predicted labels :  [0 1 0 1]

    Parameters
    -----------

    """
    def __init__(self, n_estimators=10, max_depth=-1, handle=None,
                 max_features=1.0, min_samples_split=2, n_bins=8,
                 split_algo=0, min_rows_per_node=2,
                 bootstrap=True, bootstrap_features=False,
                 type_model="classifier", verbose=False,
                 rows_sample=1.0, max_leaves=-1,
                 gdf_datatype=None):

        super(RandomForestClassifier, self).__init__(handle, verbose)
        self._impl = RandomForest_impl(n_estimators, max_depth, self.handle,
                                       max_features, min_samples_split, n_bins,
                                       split_algo, min_rows_per_node,
                                       bootstrap, bootstrap_features,
                                       type_model, verbose,
                                       rows_sample, max_leaves,
                                       gdf_datatype)

    def fit(self, X, y):

        return self._impl.fit(X, y)

    def predict(self, X):

        return self._impl.predict(X)

    def cross_validate(self, X, y):

        return self._impl.cross_validate(X, y)
