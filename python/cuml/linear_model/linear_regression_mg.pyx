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

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free


cdef extern from "glm/glm_spmg.h" namespace "ML::GLM":

    cdef void olsFitSPMG(float *h_input,
                         int n_rows,
                         int n_cols,
                         float *h_labels,
                         float *h_coef,
                         float *intercept,
                         bool fit_intercept,
                         bool normalize,
                         int *gpu_ids,
                         int n_gpus)

    cdef void olsFitSPMG(double *h_input,
                         int n_rows,
                         int n_cols,
                         double *h_labels,
                         double *h_coef,
                         double *intercept,
                         bool fit_intercept,
                         bool normalize,
                         int *gpu_ids,
                         int n_gpus)

    cdef void olsPredictSPMG(float *input,
                             int n_rows,
                             int n_cols,
                             float *h_coef,
                              float intercept,
                             float *preds,
                             int *gpu_ids,
                             int n_gpus)

    cdef void olsPredictSPMG(double *input,
                             int n_rows,
                             int n_cols,
                             double *h_coef,
                     double intercept,
                             double *preds,
                             int *gpu_ids,
                             int n_gpus)

    cdef void spmgOlsFit(float **input,
                         int *input_cols,
                         int n_rows,
                         int n_cols,
                         float **labels,
                         int *label_rows,
                         float **coef,
                         int *coef_cols,
                         float *intercept,
                         bool fit_intercept,
                         bool normalize,
                         int n_gpus)

    cdef void spmgOlsFit(double **input,
                         int *input_cols,
                         int n_rows,
                         int n_cols,
                         double **labels,
                         int *label_rows,
                         double **coef,
                         int *coef_cols,
                         double *intercept,
                         bool fit_intercept,
                         bool normalize,
                         int n_gpus)

    cdef void spmgOlsPredict(float **input,
                             int *input_cols,
                             int n_rows,
                             int n_cols,
                             float **coef,
                             int *coef_cols,
                             float intercept,
                             float **preds,
                             int *pred_cols,
                             int n_gpus)

    cdef void spmgOlsPredict(double **input,
                             int *input_cols,
                             int n_rows,
                             int n_cols,
                             double **coef,
                             int *coef_cols,
                             double intercept,
                             double **preds,
                             int *pred_cols,
                             int n_gpus)


class LinearRegressionMG:

    """
    Single Process, Multi-GPU Linear Regression

    For using with Numpy, assuming 2 GPUs:

    .. code-block:: python

        from cuml import LinearRegression
        import numpy as np


        X = np.array([
        [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 1.0, 11.0, 21.0, 31.0, 41.0, 51.0],
        [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 2.0, 12.0, 22.0, 32.0, 42.0, 52.0],
        [3.0, 13.0, 23.0, 33.0, 43.0, 53.0, 3.0, 13.0, 23.0, 33.0, 43.0, 53.0],
        [4.0, 14.0, 24.0, 34.0, 44.0, 54.0, 4.0, 14.0, 24.0, 34.0, 44.0, 54.0],
        [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 1.0, 11.0, 21.0, 31.0, 41.0, 51.0],
        [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 2.0, 12.0, 22.0, 32.0, 42.0, 52.0],
        [3.0, 13.0, 23.0, 33.0, 43.0, 53.0, 3.0, 13.0, 23.0, 33.0, 43.0, 53.0],
        [4.0, 14.0, 24.0, 34.0, 44.0, 54.0, 4.0, 14.0, 24.0, 34.0, 44.0, 54.0],
        [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 1.0, 11.0, 21.0, 31.0, 41.0, 51.0],
        [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 2.0, 12.0, 22.0, 32.0, 42.0, 52.0],
        [3.0, 13.0, 23.0, 33.0, 43.0, 53.0, 3.0, 13.0, 23.0, 33.0, 43.0, 53.0],
        [4.0, 14.0, 24.0, 34.0, 44.0, 54.0, 4.0, 14.0, 24.0, 34.0, 44.0, 54.0],
        [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 1.0, 11.0, 21.0, 31.0, 41.0, 51.0],
        [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 2.0, 12.0, 22.0, 32.0, 42.0, 52.0],
        [3.0, 13.0, 23.0, 33.0, 43.0, 53.0, 3.0, 13.0, 23.0, 33.0, 43.0, 53.0],
        [4.0, 14.0, 24.0, 34.0, 44.0, 54.0, 4.0, 14.0, 24.0, 34.0, 44.0, 54.0]
        ], dtype=np.float32)



        y = np.array([60.0, 61.0, 62.0, 63.0, 60.0, 61.0, 62.0, 63.0, 60.0,
                      61.0, 62.0, 63.0, 60.0, 61.0, 62.0, 63.0],
                     dtype=np.float32)

        lr = LinearRegression()

        res = lr.fit(X, y, gpu_ids=[0,1])


    To use with Dask, please see the LinearRegression in dask-cuml.

    """

    def __init__(self, algorithm='eig', fit_intercept=True, normalize=False):

        """
        Initializes the linear regression class.

        Parameters
        ----------
        algorithm : Type: string. 'eig' (default) and 'svd' are supported
        algorithms.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        if algorithm in ['svd', 'eig']:
            self.algo = self._get_algorithm_int(algorithm)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))

        self.intercept_value = 0.0

    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1
        }[algorithm]

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X, y, n_gpus=1, gpu_ids=[]):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        n_gpus : int
                 Number of gpus to be used for prediction. If gpu_ids parameter
                 has more than element, this parameter is ignored.

        gpu_ids: int array
                 GPU ids to be used for prediction.

        """

        if (len(gpu_ids) > 1):
            return self._fit_spmg(X, y, gpu_ids)
        elif (n_gpus > 1):
            for i in range(0, n_gpus):
                gpu_ids.append(i)
            return self._fit_spmg(X, y, gpu_ids)
        else:
            raise ValueError('Number of GPUS should be 2 or more'
                             'For single GPU, use the normal LinearRegression')

    def predict(self, X, n_gpus=1, gpu_ids=[]):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        n_gpus : int
                 Number of gpus to be used for prediction. If gpu_ids parameter
                 has more than element, this parameter is ignored.

        gpu_ids: int array
                 GPU ids to be used for prediction.

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        if (len(gpu_ids) > 1):
            return self._predict_spmg(X, gpu_ids)
        elif (n_gpus > 1):
            for i in range(0, n_gpus):
                gpu_ids.append(i)
            return self._predict_spmg(X, gpu_ids)
        else:
            raise ValueError('Number of GPUS should be 2 or more'
                             'For single GPU, use the normal LinearRegression')

    def _fit_spmg(self, X, y, gpu_ids):
        # Using numpy ctypes pointer to avoid cimport numpy for abi issues
        # Future improvement change saving this coefs as distributed in gpus

        if (not isinstance(X, np.ndarray)):
            msg = "X matrix must be a Numpy ndarray." \
                  " Dask will be supported in the next version."
            raise TypeError(msg)

        if (not isinstance(y, np.ndarray)):
            msg = "y matrix must be a Numpy ndarray." \
                  " Dask will be supported in the next version."
            raise TypeError(msg)

        n_gpus = len(gpu_ids)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        if (n_rows < 1):
            msg = "X must have a row."
            raise TypeError(msg)

        if (n_cols < 1):
            msg = "X must have a column."
            raise TypeError(msg)

        n_rows_y = y.shape[0]

        if (n_rows_y < 1):
            msg = "y must have a row."
            raise TypeError(msg)

        if (n_rows != n_rows_y):
            msg = "X and y must have a same number of rows."
            raise TypeError(msg)

        if (not np.isfortran(X)):
            X = np.array(X, order='F', dtype=X.dtype)

        if (not np.isfortran(y)):
            y = np.array(y, order='F', dtype=y.dtype)

        cdef uintptr_t X_ptr, y_ptr, gpu_ids_ptr, coef_ptr

        self.gdf_datatype = X.dtype
        self.coef_ = np.zeros(X.shape[1], dtype=X.dtype)

        X_ptr = X.ctypes.data
        y_ptr = y.ctypes.data
        gpu_id_32 = np.array(gpu_ids, dtype=np.int32)
        gpu_ids_ptr = gpu_id_32.ctypes.data
        coef_ptr = self.coef_.ctypes.data

        cdef float intercept32
        cdef double intercept64

        if self.gdf_datatype.type == np.float32:
            olsFitSPMG(<float*>X_ptr,
                       <int> n_rows,
                       <int> n_cols,
                       <float*>y_ptr,
                       <float*>coef_ptr,
                       <float*>&intercept32,
                       <bool>self.fit_intercept,
                       <bool>self.normalize,
                       <int*>gpu_ids_ptr,
                       <int>n_gpus)

            self.intercept_ = intercept32

        else:
            olsFitSPMG(<double*>X_ptr,
                       <int> n_rows,
                       <int> n_cols,
                       <double*>y_ptr,
                       <double*>coef_ptr,
                       <double*>&intercept64,
                       <bool>self.fit_intercept,
                       <bool>self.normalize,
                       <int*>gpu_ids_ptr,
                       <int>n_gpus)

            self.intercept_ = intercept64

        return self

    def _predict_spmg(self, X, gpu_ids):

        n_gpus = len(gpu_ids)

        if (not np.isfortran(X)):
            X = np.array(X, order='F', dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        if (n_rows < 1):
            msg = "X must have a row."
            raise TypeError(msg)

        if (n_cols < 1):
            msg = "X must have a column."
            raise TypeError(msg)

        pred = np.zeros(n_rows, dtype=X.dtype)

        cdef uintptr_t X_ptr, pred_ptr, gpu_ids_ptr, coef_ptr

        self.gdf_datatype = X.dtype

        X_ptr = X.ctypes.data
        pred_ptr = pred.ctypes.data
        gpu_id_32 = np.array(gpu_ids, dtype=np.int32)
        gpu_ids_ptr = gpu_id_32.ctypes.data
        coef_ptr = self.coef_.ctypes.data

        if self.gdf_datatype.type == np.float32:
            olsPredictSPMG(<float*>X_ptr,
                           <int> n_rows,
                           <int> n_cols,
                           <float*>coef_ptr,
                           <float>self.intercept_,
                           <float*>pred_ptr,
                           <int*>gpu_ids_ptr,
                           <int>n_gpus)

        else:
            olsPredictSPMG(<double*>X_ptr,
                           <int> n_rows,
                           <int> n_cols,
                           <double*>coef_ptr,
                           <double>self.intercept_,
                           <double*>pred_ptr,
                           <int*>gpu_ids_ptr,
                           <int>n_gpus)

        return pred

    def _fit_mg(self, alloc_info, params):

        self.fit_intercept = params['fit_intercept']
        self.normalize = params['normalize']

        if alloc_info[0][0]['typestr'] == '<f8':
            dtype = 'double'
        else:
            dtype = 'single'

        cdef int n_rows
        cdef int n_cols = 0

        n_rows = alloc_info[0][0]["shape"][0]

        cdef n_inputs = len(alloc_info)

        cdef int* input_cols
        cdef int* label_rows
        cdef int* coef_cols
        input_cols = <int*>malloc(len(alloc_info*sizeof(int)))
        label_rows = <int*>malloc(len(alloc_info*sizeof(int)))
        coef_cols = <int*>malloc(len(alloc_info*sizeof(int)))

        cdef uintptr_t input_ptr
        n_allocs = len(alloc_info)

        cdef float** input32
        cdef float** labels32
        cdef float** coef32
        cdef double** input64
        cdef double** labels64
        cdef double** coef64

        cdef float intercept_f32
        cdef double  intercept_f64
        intercept_f32 = 0
        intercept_f64 = 0

        if dtype == "single":
            input32 = <float**>malloc(len(alloc_info)*sizeof(float*))
            labels32 = <float**>malloc(len(alloc_info)*sizeof(float*))
            coef32 = <float**>malloc(len(alloc_info)*sizeof(float*))

            idx = 0
            for ary in alloc_info:
                input_ptr = ary[0]['data'][0]
                input32[idx] = <float*>input_ptr
                input_cols[idx] = ary[0]["shape"][1]
                n_cols = n_cols + ary[0]["shape"][1]

                input_ptr = ary[1]['data'][0]
                labels32[idx] = <float*>input_ptr
                label_rows[idx] = ary[1]["shape"][0]

                input_ptr = ary[2]['data'][0]
                coef32[idx] = <float*>input_ptr
                coef_cols[idx] = ary[2]["shape"][0]

                idx = idx + 1

            spmgOlsFit(<float**> input32,
                       <int*> input_cols,
                       <int> n_rows,
                       <int> n_cols,
                       <float**> labels32,
                       <int*> label_rows,
                       <float**> coef32,
                       <int*> coef_cols,
                       <float*> &intercept_f32,
                       <bool> self.fit_intercept,
                       <bool> self.normalize,
                       <int> n_allocs)

            return intercept_f32

        else:

            input64 = <double**>malloc(len(alloc_info)*sizeof(double*))
            labels64 = <double**>malloc(len(alloc_info)*sizeof(double*))
            coef64 = <double**>malloc(len(alloc_info)*sizeof(double*))

            idx = 0
            for ary in alloc_info:
                input_ptr = ary[0]['data'][0]
                input64[idx] = <double*>input_ptr
                input_cols[idx] = ary[0]["shape"][1]
                n_cols = n_cols + ary[0]["shape"][1]

                input_ptr = ary[1]['data'][0]
                labels64[idx] = <double*>input_ptr
                label_rows[idx] = ary[1]["shape"][0]

                input_ptr = ary[2]['data'][0]
                coef64[idx] = <double*>input_ptr
                coef_cols[idx] = ary[2]["shape"][0]

                idx = idx + 1

            spmgOlsFit(<double**> input64,
                       <int*> input_cols,
                       <int> n_rows,
                       <int> n_cols,
                       <double**> labels64,
                       <int*> label_rows,
                       <double**> coef64,
                       <int*> coef_cols,
                       <double*> &intercept_f64,
                       <bool> self.fit_intercept,
                       <bool> self.normalize,
                       <int> n_allocs)

            return intercept_f64

    def _predict_mg(self, alloc_info, intercept, params):

        self.fit_intercept = params['fit_intercept']
        self.normalize = params['normalize']

        if alloc_info[0][0]['typestr'] == '<f8':
            dtype = 'double'
        else:
            dtype = 'single'

        cdef int n_rows
        cdef int n_cols = 0

        n_rows = alloc_info[0][0]["shape"][0]

        cdef n_inputs = len(alloc_info)

        cdef float** input32
        cdef float** pred32
        cdef float** coef32
        cdef double** input64
        cdef double** pred64
        cdef double** coef64

        cdef int* input_cols
        cdef int* pred_rows
        cdef int* coef_cols
        input_cols = <int*>malloc(len(alloc_info)*sizeof(int))
        pred_rows = <int*>malloc(len(alloc_info)*sizeof(int))
        coef_cols = <int*>malloc(len(alloc_info)*sizeof(int))

        cdef uintptr_t input_ptr
        n_allocs = len(alloc_info)

        if dtype == "single":

            input32 = <float**>malloc(len(alloc_info)*sizeof(float*))
            pred32 = <float**>malloc(len(alloc_info)*sizeof(float*))
            coef32 = <float**>malloc(len(alloc_info)*sizeof(float*))

            idx = 0
            for ary in alloc_info:
                input_ptr = ary[0]['data'][0]
                input32[idx] = <float*>input_ptr
                input_cols[idx] = ary[0]["shape"][1]
                n_cols = n_cols + ary[0]["shape"][1]

                input_ptr = ary[1]['data'][0]
                coef32[idx] = <float*>input_ptr
                coef_cols[idx] = ary[1]["shape"][0]

                input_ptr = ary[2]['data'][0]
                pred32[idx] = <float*>input_ptr
                pred_rows[idx] = ary[2]["shape"][0]

                idx = idx + 1

            spmgOlsPredict(<float**>input32,
                           <int*>input_cols,
                           <int> n_rows,
                           <int> n_cols,
                           <float**>coef32,
                           <int*>coef_cols,
                           <float>intercept,
                           <float**>pred32,
                           <int*>pred_rows,
                           <int> n_allocs)

        else:

            input64 = <double**>malloc(len(alloc_info)*sizeof(double*))
            pred64 = <double**>malloc(len(alloc_info)*sizeof(double*))
            coef64 = <double**>malloc(len(alloc_info)*sizeof(double*))

            idx = 0
            for ary in alloc_info:
                input_ptr = ary[0]['data'][0]
                input64[idx] = <double*>input_ptr
                input_cols[idx] = ary[0]["shape"][1]
                n_cols = n_cols + ary[0]["shape"][1]

                input_ptr = ary[1]['data'][0]
                coef64[idx] = <double*>input_ptr
                coef_cols[idx] = ary[1]["shape"][0]

                input_ptr = ary[2]['data'][0]
                pred64[idx] = <double*>input_ptr
                pred_rows[idx] = ary[2]["shape"][0]

                idx = idx + 1

            spmgOlsPredict(<double**>input64,
                           <int*>input_cols,
                           <int> n_rows,
                           <int> n_cols,
                           <double**>coef64,
                           <int*>coef_cols,
                           <double>intercept,
                           <double**>pred64,
                           <int*>pred_rows,
                           <int> n_allocs)
