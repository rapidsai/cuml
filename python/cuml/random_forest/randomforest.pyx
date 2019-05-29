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
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr

cdef extern from "randomforest/randomforest.h" namespace "ML":

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
        pass

    cdef void fit(cumlHandle& handle,
                  rfClassifier[float]*,
                  float *,
                  int ,
                  int ,
                  int *,
                  int)

    cdef void fit(cumlHandle& handle,
                  rfClassifier[double]*,
                  double *,
                  int ,
                  int ,
                  int *,
                  int)

    cdef void predict(cumlHandle& handle,
                      rfClassifier[float]*,
                      float *,
                      int ,
                      int ,
                      int *,
                      bool)

    cdef void predict(cumlHandle& handle,
                      rfClassifier[double]*,
                      double *,
                      int ,
                      int ,
                      int *,
                      bool)

    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   rfClassifier[float] * , float * , int * ,
				   int , int , int * , bool);
    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   rfClassifier[double] *, double * , int * ,
				   int , int , int * , bool);

class RandomForest():
    """
    Description and example code

    """
    
    # min_rows_per_node in cuml = min_samples_split in sklearn
    def __init__(self, n_estimators=10, max_depth=None, max_features=None, min_samples_split=None, bootstrap=True, type="classifier"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.type = self._get_type(type)
        self.bootstrap = bootstrap
        self.verbose = False

    def _get_type(self, type):
        if type == "classifier":
            self.type = 1
        else:
            self.type = 0

    def fit(self, X, y):

        cdef uintptr_t X_ptr, y_ptr

        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='F')
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        #X_m, X_ptr, n_rows, n_cols, dtype = input_to_array(X)
        #y_m, y_ptr, _, _, _ = input_to_array(y)

        X_ptr = self._get_dev_array_ptr(X_m)


        if (isinstance(y, cudf.Series)):
            y_ptr = self._get_cudf_column_ptr(y)
        elif (isinstance(y, np.ndarray)):
            y_m = cuda.to_device(y)
            y_ptr = self._get_dev_array_ptr(y_m)
        else:
            msg = "y vector must be a cuDF series or Numpy ndarray"
            raise TypeError(msg)

        cdef cumlHandle* handle_ = <cumlHandle*> <size_t> self.handle.getHandle()

        n_unique_labels = 10

        cdef rfClassifier[float] *rf_classifier32
        cdef rfClassifier[double] *rf_classifier64

        if self.gdf_datatype.type == np.float32:
            fit(handle_[0],
                rf_classifier32,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> n_unique_labels)
        else:
            fit(handle_[0],
                rf_classifier64,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> n_unique_labels)

        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self


    def predict(self, X):

        cdef uintptr_t X_ptr
        #X_m, X_ptr, n_rows, n_cols, dtype = input_to_array(X)
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='F')
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        X_ptr = self._get_dev_array_ptr(X_m)

        preds = cudf.Series(np.zeros(self.n_rows, dtype=self.gdf_datatype))
        cdef uintptr_t preds_ptr = self._get_cudf_column_ptr(preds).value

        cdef cumlHandle* handle_ = <cumlHandle*> <size_t> self.handle.getHandle()


        cdef rfClassifier[float] *rf_classifier_pred32
        cdef rfClassifier[double] *rf_classifier_pred64

        if self.gdf_datatype.type == np.float32:
            predict(handle_[0],
                    rf_classifier_pred32,
                    <float*> X_ptr,
                    <int> self.n_rows,
                    <int> self.n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)

        elif self.gdf_datatype.type == np.float64:
            predict(handle_[0],
                    rf_classifier_pred64,
                    <double*> X_ptr,
                    <int> self.n_rows,
                    <int> self.n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)

        else:
            raise TypeError("supports only float32 and float64 input, but input of type '%s' passed." % (str(self.gdf_datatype.type)))

        self.handle.sync()
        del(X_m)
        return preds
