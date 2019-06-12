import ctypes
import cudf
import numpy as np
import warnings
import pdb
from numba import cuda
from cuml import numba_utils
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
cimport cuml.common.handle
cimport cuml.common.cuda
# from cuml.utils.input_utils import get_cudf_column_ptr, get_dev_array_ptr


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
                  int)

    cdef void fit(cumlHandle& handle,
                  rfClassifier[double] *,
                  double *,
                  int,
                  int,
                  int *,
                  int)

    cdef void predict(cumlHandle& handle,
                      rfClassifier[float] *,
                      float *,
                      int,
                      int,
                      int *,
                      bool)

    cdef void predict(cumlHandle& handle,
                      rfClassifier[double] *,
                      double *,
                      int,
                      int,
                      int *,
                      bool)

    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   rfClassifier[float] *, float *, int *,
                                   int, int, int *, bool)
    cdef RF_metrics cross_validate(cumlHandle& handle,
                                   rfClassifier[double] *, double *, int *,
                                   int, int, int *, bool)

    cdef RF_params set_rf_class_obj(int, int, float,
                                    int, int, int,
                                    bool, bool, int, int)


cdef class RandomForest_impl():
    """
    Description and example code
    split_algo = 0 for HIST, 1 for GLOBAL_QUANTILE and 3 for SPLIT_ALGO_END
    """
    cdef uintptr_t X_ptr, y_ptr
    cpdef object handle
    cdef rfClassifier[float] *rf_classifier32
    cdef rfClassifier[double] *rf_classifier64
    cdef uintptr_t preds_ptr
    cdef object X_m
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
    cdef object rows_sample
    cdef object max_leaves
    cdef object gdf_datatype
    cdef object n_rows
    cdef object n_cols
    cdef object preds
    cdef object stats

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
        self.n_rows = None
        self.n_cols = None
        self.preds = None

    def _get_type(self, type_model):
        if type_model == "classifier":
            self.type_model = 1
        else:
            self.type_model = 0

    def _get_dev_array_ptr(self, obj):
        """
        Get ctype pointer of a numba style device array
        """
        return obj.device_ctypes_pointer.value

    def _get_cudf_column_ptr(self, col):
        """
        Get ctype pointer of a cudf column
        """
        return cudf.bindings.cudf_cpp.get_column_data_ptr(col._column)

    def fit(self, X, y):

        cdef uintptr_t X_ptr, y_ptr

        if self.rf_classifier32 != NULL:
            del self.rf_classifier32

        if self.rf_classifier64 != NULL:
            del self.rf_classifier64

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

        '''
        X_m, X_ptr, n_rows, n_cols, dtype = input_to_array(X)
        y_m, y_ptr, _, _, _ = input_to_array(y)
        '''
        X_ptr = self._get_dev_array_ptr(X_m)

        if (isinstance(y, cudf.Series)):
            y_ptr = self._get_cudf_column_ptr(y)
        elif (isinstance(y, np.ndarray)):
            y_m = cuda.to_device(y)
            y_ptr = self._get_dev_array_ptr(y_m)
        else:
            msg = "y vector must be a cuDF series or Numpy ndarray"
            raise TypeError(msg)

        cdef cumlHandle * handle_ =\
            <cumlHandle *> <size_t> self.handle.getHandle()

        unique_labels = np.shape(np.unique(y))[0]

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

        if self.gdf_datatype.type == np.float32:
            fit(handle_[0],
                self.rf_classifier32,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> unique_labels)
        else:
            fit(handle_[0],
                self.rf_classifier64,
                <double *> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int *> y_ptr,
                <int> unique_labels)

        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self

    def predict(self, X):

        cdef uintptr_t X_ptr
        X_ptr = X.ctypes.data
        self.n_rows, self.n_cols = np.shape(X)
        preds = np.zeros(self.n_rows,
                         dtype=np.int32)
        cdef uintptr_t preds_ptr = preds.ctypes.data
        cdef cumlHandle * handle_ =\
            <cumlHandle *> <size_t> self.handle.getHandle()

        if self.gdf_datatype.type == np.float32:
            predict(handle_[0],
                    self.rf_classifier32,
                    <float *> X_ptr,
                    <int> self.n_rows,
                    <int> self.n_cols,
                    <int *> preds_ptr,
                    <bool> self.verbose)

        elif self.gdf_datatype.type == np.float64:
            predict(handle_[0],
                    self.rf_classifier64,
                    <double *> X_ptr,
                    <int> self.n_rows,
                    <int> self.n_cols,
                    <int *> preds_ptr,
                    <bool> self.verbose)

        else:
            raise TypeError("supports only float32 and float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.gdf_datatype.type)))

        self.handle.sync()
        return preds

    def cross_validate(self, X, y):

        cdef uintptr_t X_ptr, y_ptr
        X_ptr = X.ctypes.data
        y_ptr = y.ctypes.data
        self.n_rows, self.n_cols = np.shape(X)
        preds = np.zeros(self.n_rows,
                         dtype=np.int32)
        cdef uintptr_t preds_ptr = (preds).ctypes.data

        cdef cumlHandle * handle_ =\
            <cumlHandle *> <size_t> self.handle.getHandle()

        if self.gdf_datatype.type == np.float32:
            self.stats = cross_validate(handle_[0],
                                        self.rf_classifier32,
                                        <float *> X_ptr,
                                        <int *> y_ptr,
                                        <int> self.n_rows,
                                        <int> self.n_cols,
                                        <int *> preds_ptr,
                                        <bool> self.verbose)

        elif self.gdf_datatype.type == np.float64:
            self.stats = cross_validate(handle_[0],
                                        self.rf_classifier64,
                                        <double *> X_ptr,
                                        <int *> y_ptr,
                                        <int> self.n_rows,
                                        <int> self.n_cols,
                                        <int *> preds_ptr,
                                        <bool> self.verbose)

        self.handle.sync()
        return self.stats


class RandomForest(Base):

    def __init__(self, n_estimators=10, max_depth=-1, handle=None,
                 max_features=1.0, min_samples_split=2, n_bins=8,
                 split_algo=0, min_rows_per_node=2,
                 bootstrap=True, bootstrap_features=False,
                 type_model="classifier", verbose=False,
                 rows_sample=1.0, max_leaves=-1,
                 gdf_datatype=None):

        super(RandomForest, self).__init__(handle, verbose)
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
