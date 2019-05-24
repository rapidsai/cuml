import ctypes
import cudf
import numpy as np
import warnings
from numba import cuda
from cuml import numba_utils
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_array

cdef extern from "randomforest/randomforest.hpp" namespace "ML":

    cdef struct RF_metrics:
        float accuracy

    cdef enum RF_type:
        CLASSIFICATION,
        REGRESSION

    cdef struct RF_params:
        bool bootstrap,
        bool bootstrap_features,
        int n_trees,
        float rows_sample

    cdef cppclass rf:
        pass

    cdef cppclass rfClassifier[T]:
        rfClassifier()
        rfClassifier(RF_params *)
        void fit(const cumlHandle& user_handle, T * input, int n_rows,
		 int n_cols, int * labels, int n_unique_labels);
        void predict(const cumlHandle& user_handle, float * input, 
		     int n_rows, int n_cols, int * predictions,
		     bool verbose=false) const;
        RF_metrics cross_validate(const cumlHandle& user_handle, const T * input,
				  const int * ref_labels, int n_rows, 
				  int n_cols, int * predictions, 
				  bool verbose=false) const;

    cdef void fit(cumlHandle& handle,
                  float *rf_classifier,
                  float *input,
                  int n_rows,
                  int n_cols,
                  int *labels)

    cdef void fit(cumlHandle& handle,
                  double *rf_classifier,
                  double *input,
                  int n_rows,
                  int n_cols,
                  int *labels)

    cdef void predict(cumlHandle& handle,
                      float *rf_classifier,
                      float *input,
                      int n_rows,
                      int n_cols,
                      int *predictions,
                      bool verbose)

    cdef void predict(cumlHandle& handle,
                      double *rf_classifier,
                      double *input,
                      int n_rows,
                      int n_cols,
                      int *predictions,
                      bool verbose)

    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   float * rf_classifier, float * input, int * ref_labels,
				   int n_rows, int n_cols, int * predictions, bool verbos);
    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   double * rf_classifier, double * input, int * ref_labels,
				   int n_rows, int n_cols, int * predictions, bool verbose);

class RandomForest():
    """
    Description and example code

    """
    
    # min_rows_per_node in cuml = min_samples_split in sklearn
    def __init__(self, n_estimators=10, max_depth=None, 
		 max_features=None, min_samples_split=None, 
		 bootstrap=True, type="classifier"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.type = self._get_type(type)
        self.bootstrap = bootstrap

    def _get_type(self, type):
        if type == "classifier":
            self.type = 1
        else:
            self.type = 0

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X, y):

        cdef uintptr_t X_ptr, y_ptr
        X_m, X_ptr, n_rows, n_cols, dtype = input_to_array(X)
        y_m, y_ptr, _, _, _ = input_to_array(y)

        cdef cumlHandle* handle_ = <cumlHandle*> <size_t> self.handle.getHandle()

        cdef rfClassifier rf_classifier

        if self.gdf_datatype.type == np.float32:
            fit(handle_[0],
                <float*> rf_classifier,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr)
        else:
            fit(handle_[0],
                <float*> rf_classifier,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr)

        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self


    def predict(self, X):

        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, dtype = input_to_array(X)

        preds = cudf.Series(np.zeros(n_rows, dtype=dtype))
        cdef uintptr_t preds_ptr = self._get_cudf_column_ptr(preds)

        cdef cumlHandle* handle_ = <cumlHandle*> <size_t> self.handle.getHandle()

        cdef rfClassifier rf_classifier

        if self.gdf_datatype.type == np.float32:
            fit(handle_[0],
                <float*> rf_classifier,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> preds_ptr)

        elif self.gdf_datatype.type == np.float64:
            fit(handle_[0],
                <double*> rf_classifier,
                <double*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> preds_ptr)

        else:
            raise TypeError("supports only float32 and float64 input, "
			    "but input of type '%s' passed." % (str(self.gdf_datatype.type)))

        self.handle.sync()
        del(X_m)
        return preds
