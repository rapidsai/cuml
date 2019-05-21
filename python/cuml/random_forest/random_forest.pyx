import ctypes
import cudf
import numpy as np
import warnings
from numba import cuda
from cuml import numba_utils
from cuml.common.handle cimport cumlHandle
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base


cdef extern from "randomforest/randomforest.hpp" namespace "ML":



    struct RF_metrics:
        float accuracy,

    enum RF_type:
        CLASSIFICATION,
        REGRESSION

    struct RF_params:
        bool bootstrap,
        bool bootstrap_features,
        int n_trees,
        float rows_sample
        
    cdef cppclass rf:
        pass
                    

            
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
				   int n_rows, int n_cols, int * predictions, bool verbose=false);
    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   double * rf_classifier, double * input, int * ref_labels,
				   int n_rows, int n_cols, int * predictions, bool verbose=false);


cdef extern from "randomforest/randomforest.hpp" namespace "ML::rfClassifier":
    cdef void fit(PASS)
    cdef void predict(PASS)
    RF_metric cross_validate(PASS)                  
        


class Randomforest(Base):
    """
    Description and example code

    """
    
    # min_rows_per_node in cuml = min_samples_split in sklearn
    # max_leaves
    def __init__(self, n_estimators=25, max_depth=None, max_features=None, min_rows_per_node=None, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_rows_per_node = min_rows_per_node
        self.bootstrap = bootstrap

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X):

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

        cdef cumlHandle* handle_ = <cumlHandle*> <size_t> self.handle.getHandle()
        self.labels_ = cudf.Series(np.zeros(self.n_rows, dtype=np.int32))
        self.labels_array = self.labels_._column._data.to_gpu_array()
        cdef uintptr_t labels_ptr = self._get_ctype_ptr(self.labels_array)

        cdef rfClassifier rf_classifier

        if self.gdf_datatype.type == np.float32:
            fit(handle_[0],
                <float*> rf_classifier,
                <float*> input_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> labels_ptr)
        else:
            fit(handle_[0],
                <float*> rf_classifier,
                <float*> input_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> labels_ptr)

        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        return self


    # n_unique_labels is found while running the preprocess_labels function 

    def predict(self, X):
       
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

        cdef cumlHandle* handle_ = <cumlHandle*> <size_t> self.handle.getHandle()
        clust_mat = numba_utils.row_matrix(self.cluster_centers_)
        cdef uintptr_t cluster_centers_ptr = self._get_ctype_ptr(clust_mat)

        self.labels_ = cudf.Series(np.zeros(self.n_rows, dtype=np.int32))
        cdef uintptr_t labels_ptr = self._get_column_ptr(self.labels_)

        cdef rfClassifier rf_classifier

        if self.gdf_datatype.type == np.float32:
            fit(handle_[0],
                <float*> rf_classifier,
                <float*> input_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> labels_ptr)

        elif self.gdf_datatype.type == np.float64:
            fit(handle_[0],
                <double*> rf_classifier,
                <double*> input_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> labels_ptr)

        else:
            raise TypeError("supports only float32 and float64 input, but input of type '%s' passed." % (str(self.gdf_datatype.type)))

        self.handle.sync()
        del(X_m)
        del(clust_mat)
        return self.labels_
