# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

# distutils: language = c++

import cudf
import cupy as cp
import numpy as np

from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.array_sparse import SparseCumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.mixins import FMajorInputTagMixin
from cuml.common.sparse_utils import is_sparse
from cuml.metrics import accuracy_score


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

    void qnFit(
        handle_t& cuml_handle,
        float *X,
        bool X_col_major,
        float *y,
        int N,
        int D,
        int C,
        bool fit_intercept,
        float l1,
        float l2,
        int max_iter,
        float grad_tol,
        float change_tol,
        int linesearch_max_iter,
        int lbfgs_memory,
        int verbosity,
        float *w0,
        float *f,
        int *num_iters,
        int loss_type,
        float *sample_weight) except +

    void qnFit(
        handle_t& cuml_handle,
        double *X,
        bool X_col_major,
        double *y,
        int N,
        int D,
        int C,
        bool fit_intercept,
        double l1,
        double l2,
        int max_iter,
        double grad_tol,
        double change_tol,
        int linesearch_max_iter,
        int lbfgs_memory,
        int verbosity,
        double *w0,
        double *f,
        int *num_iters,
        int loss_type,
        double *sample_weight) except +

    void qnFitSparse(
        handle_t& cuml_handle,
        float *X_values,
        int *X_cols,
        int *X_row_ids,
        int X_nnz,
        float *y,
        int N,
        int D,
        int C,
        bool fit_intercept,
        float l1,
        float l2,
        int max_iter,
        float grad_tol,
        float change_tol,
        int linesearch_max_iter,
        int lbfgs_memory,
        int verbosity,
        float *w0,
        float *f,
        int *num_iters,
        int loss_type,
        float *sample_weight) except +

    void qnFitSparse(
        handle_t& cuml_handle,
        double *X_values,
        int *X_cols,
        int *X_row_ids,
        int X_nnz,
        double *y,
        int N,
        int D,
        int C,
        bool fit_intercept,
        double l1,
        double l2,
        int max_iter,
        double grad_tol,
        double change_tol,
        int linesearch_max_iter,
        int lbfgs_memory,
        int verbosity,
        double *w0,
        double *f,
        int *num_iters,
        int loss_type,
        double *sample_weight) except +

    void qnDecisionFunction(
        handle_t& cuml_handle,
        float *X,
        bool X_col_major,
        int N,
        int D,
        int C,
        bool fit_intercept,
        float *params,
        int loss_type,
        float *scores) except +

    void qnDecisionFunction(
        handle_t& cuml_handle,
        double *X,
        bool X_col_major,
        int N,
        int D,
        int C,
        bool fit_intercept,
        double *params,
        int loss_type,
        double *scores) except +

    void qnDecisionFunctionSparse(
        handle_t& cuml_handle,
        float *X_values,
        int *X_cols,
        int *X_row_ids,
        int X_nnz,
        int N,
        int D,
        int C,
        bool fit_intercept,
        float *params,
        int loss_type,
        float *scores) except +

    void qnDecisionFunctionSparse(
        handle_t& cuml_handle,
        double *X_values,
        int *X_cols,
        int *X_row_ids,
        int X_nnz,
        int N,
        int D,
        int C,
        bool fit_intercept,
        double *params,
        int loss_type,
        double *scores) except +

    void qnPredict(
        handle_t& cuml_handle,
        float *X,
        bool X_col_major,
        int N,
        int D,
        int C,
        bool fit_intercept,
        float *params,
        int loss_type,
        float *preds) except +

    void qnPredict(
        handle_t& cuml_handle,
        double *X,
        bool X_col_major,
        int N,
        int D,
        int C,
        bool fit_intercept,
        double *params,
        int loss_type,
        double *preds) except +

    void qnPredictSparse(
        handle_t& cuml_handle,
        float *X_values,
        int *X_cols,
        int *X_row_ids,
        int X_nnz,
        int N,
        int D,
        int C,
        bool fit_intercept,
        float *params,
        int loss_type,
        float *preds) except +

    void qnPredictSparse(
        handle_t& cuml_handle,
        double *X_values,
        int *X_cols,
        int *X_row_ids,
        int X_nnz,
        int N,
        int D,
        int C,
        bool fit_intercept,
        double *params,
        int loss_type,
        double *preds) except +


class QN(Base,
         FMajorInputTagMixin):
    """
    Quasi-Newton methods are used to either find zeroes or local maxima
    and minima of functions, and used by this class to optimize a cost
    function.

    Two algorithms are implemented underneath cuML's QN class, and which one
    is executed depends on the following rule:

      * Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
        regularization

      * Limited Memory BFGS (L-BFGS) otherwise.

    cuML's QN class can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or __cuda_array_interface__ compliant).

    Examples
    --------
    .. code-block:: python

        import cudf
        import numpy as np

        # Both import methods supported
        # from cuml import QN
        from cuml.solvers import QN

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series( np.array([0.0, 0.0, 1.0, 1.0], dtype = np.float32) )

        solver = QN()
        solver.fit(X,y)

        # Note: for now, the coefficients also include the intercept in the
        # last position if fit_intercept=True
        print("Coefficients:")
        print(solver.coef_)
        print("Intercept:")
        print(solver.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([1,5], dtype = np.float32)
        X_new['col2'] = np.array([2,5], dtype = np.float32)

        preds = solver.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python

        Coefficients:
                    10.647417
                    0.3267412
                    -17.158297
        Intercept:
                    -17.158297
        Predictions:
                    0    0.0
                    1    1.0

    Parameters
    -----------
    loss: 'sigmoid', 'softmax', 'squared_loss' (default = 'squared_loss')
        'sigmoid' loss used for single class logistic regression
        'softmax' loss used for multiclass logistic regression
        'normal' used for normal/square loss
    fit_intercept: boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    l1_strength: float (default = 0.0)
        l1 regularization strength (if non-zero, will run OWL-QN, else L-BFGS).
        Note, that as in Scikit-learn, the bias will not be regularized.
    l2_strength: float (default = 0.0)
        l2 regularization strength. Note, that as in Scikit-learn, the bias
        will not be regularized.
    max_iter: int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    tol: float (default = 1e-4)
        The training process will stop if

        `norm(current_loss_grad, inf) <= tol * max(current_loss, tol)`.

        This differs slightly from the `gtol`-controlled stopping condition in
        `scipy.optimize.minimize(method=’L-BFGS-B’)
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_:

        `norm(current_loss_projected_grad, inf) <= gtol`.

        Note, `sklearn.LogisticRegression()
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
        uses the sum of softmax/logistic loss over the input data, whereas cuML
        uses the average. As a result, Scikit-learn's loss is usually
        `sample_size` times larger than cuML's.
        To account for the differences you may divide the `tol` by the sample
        size; this would ensure that the cuML solver does not stop earlier than
        the Scikit-learn solver.
    delta: Optional[float] (default = None)
        The training process will stop if

        `abs(current_loss - previous_loss) <= delta * max(current_loss, tol)`.

        When `None`, it's set to `tol * 0.01`; when `0`, the check is disabled.
        Given the current step `k`, parameter `previous_loss` here is the loss
        at the step `k - p`, where `p` is a small positive integer set
        internally.

        Note, this parameter corresponds to `ftol` in
        `scipy.optimize.minimize(method=’L-BFGS-B’)
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_,
        which is set by default to a miniscule `2.2e-9` and is not exposed in
        `sklearn.LogisticRegression()
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
        This condition is meant to protect the solver against doing vanishingly
        small linesearch steps or zigzagging.
        You may choose to set `delta = 0` to make sure the cuML solver does
        not stop earlier than the Scikit-learn solver.
    linesearch_max_iter: int (default = 50)
        Max number of linesearch iterations per outer iteration of the
        algorithm.
    lbfgs_memory: int (default = 5)
        Rank of the lbfgs inverse-Hessian approximation. Method will use
        O(lbfgs_memory * D) memory.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    -----------
    coef_ : array, shape (n_classes, n_features)
        The estimated coefficients for the linear regression model.
        Note: shape is (n_classes, n_features + 1) if fit_intercept = True.
    intercept_ : array (n_classes, 1)
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    ------
       This class contains implementations of two popular Quasi-Newton methods:

         - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal,
           Wright - Numerical Optimization (1999)]

         - `Orthant-wise limited-memory quasi-newton (OWL-QN)
           [Andrew, Gao - ICML 2007]
           <https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/>`_
    """

    _coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    def __init__(self, *, loss='sigmoid', fit_intercept=True,
                 l1_strength=0.0, l2_strength=0.0, max_iter=1000, tol=1e-4,
                 delta=None, linesearch_max_iter=50, lbfgs_memory=5,
                 verbose=False, handle=None, output_type=None,
                 warm_start=False):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self.fit_intercept = fit_intercept
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.num_iter = 0
        self._coef_ = None
        self.intercept_ = None
        self.warm_start = warm_start

        if loss not in ['sigmoid', 'softmax', 'normal']:
            raise ValueError("loss " + str(loss) + " not supported.")

        self.loss = loss

    def _get_loss_int(self, loss):
        return {
            'sigmoid': 0,
            'softmax': 2,
            'normal': 1
        }[loss]

    @property
    @cuml.internals.api_base_return_array_skipall
    def coef_(self):
        if self.fit_intercept:
            return self._coef_[0:-1]
        else:
            return self._coef_

    @generate_docstring(X='dense_sparse')
    def fit(self, X, y, sample_weight=None, convert_dtype=False) -> "QN":
        """
        Fit the model with X and y.

        """
        sparse_input = is_sparse(X)
        # Handle sparse inputs
        if sparse_input:
            X_m = SparseCumlArray(X, convert_index=np.int32)
            n_rows, self.n_cols = X_m.shape
            self.dtype = X_m.dtype

        # Handle dense inputs
        else:
            X_m, n_rows, self.n_cols, self.dtype = input_to_cuml_array(
                X, check_dtype=[np.float32, np.float64], order='K'
            )

        y_m, lab_rows, _, _ = input_to_cuml_array(
            y, check_dtype=self.dtype,
            convert_to_dtype=(self.dtype if convert_dtype else None),
            check_rows=n_rows, check_cols=1
        )
        cdef uintptr_t y_ptr = y_m.ptr

        self._num_classes = len(cp.unique(y_m))

        cdef uintptr_t sample_weight_ptr = 0
        if sample_weight is not None:
            sample_weight, _, _, _ = \
                input_to_cuml_array(sample_weight,
                                    check_dtype=self.dtype,
                                    check_rows=n_rows, check_cols=1,
                                    convert_to_dtype=(self.dtype
                                                      if convert_dtype
                                                      else None))
            sample_weight_ptr = sample_weight.ptr

        self.loss_type = self._get_loss_int(self.loss)
        if self.loss_type != 2 and self._num_classes > 2:
            raise ValueError("Only softmax (multinomial) loss supports more"
                             "than 2 classes.")

        if self.loss_type == 2 and self._num_classes <= 2:
            raise ValueError("Two classes or less cannot be trained"
                             "with softmax (multinomial).")

        if self.loss_type == 0:
            self._num_classes_dim = self._num_classes - 1
        else:
            self._num_classes_dim = self._num_classes

        if self.fit_intercept:
            coef_size = (self.n_cols + 1, self._num_classes_dim)
        else:
            coef_size = (self.n_cols, self._num_classes_dim)

        if self._coef_ is None or not self.warm_start:
            self._coef_ = CumlArray.zeros(
                coef_size, dtype=self.dtype, order='C')

        cdef uintptr_t coef_ptr = self._coef_.ptr

        cdef float objective32
        cdef double objective64
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef int num_iters

        delta = self.delta if self.delta is not None else (self.tol * 0.01)

        if self.dtype == np.float32:
            if sparse_input:
                qnFitSparse(
                    handle_[0],
                    <float*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <float*> y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <float> self.l1_strength,
                    <float> self.l2_strength,
                    <int> self.max_iter,
                    <float> self.tol,
                    <float> delta,
                    <int> self.linesearch_max_iter,
                    <int> self.lbfgs_memory,
                    <int> self.verbose,
                    <float*> coef_ptr,
                    <float*> &objective32,
                    <int*> &num_iters,
                    <int> self.loss_type,
                    <float*> sample_weight_ptr)

            else:
                qnFit(
                    handle_[0],
                    <float*><uintptr_t> X_m.ptr,
                    <bool> __is_col_major(X_m),
                    <float*> y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <float> self.l1_strength,
                    <float> self.l2_strength,
                    <int> self.max_iter,
                    <float> self.tol,
                    <float> delta,
                    <int> self.linesearch_max_iter,
                    <int> self.lbfgs_memory,
                    <int> self.verbose,
                    <float*> coef_ptr,
                    <float*> &objective32,
                    <int*> &num_iters,
                    <int> self.loss_type,
                    <float*> sample_weight_ptr)

            self.objective = objective32

        else:
            if sparse_input:
                qnFitSparse(
                    handle_[0],
                    <double*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <double*> y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <double> self.l1_strength,
                    <double> self.l2_strength,
                    <int> self.max_iter,
                    <double> self.tol,
                    <double> delta,
                    <int> self.linesearch_max_iter,
                    <int> self.lbfgs_memory,
                    <int> self.verbose,
                    <double*> coef_ptr,
                    <double*> &objective64,
                    <int*> &num_iters,
                    <int> self.loss_type,
                    <double*> sample_weight_ptr)

            else:
                qnFit(
                    handle_[0],
                    <double*><uintptr_t> X_m.ptr,
                    <bool> __is_col_major(X_m),
                    <double*> y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <double> self.l1_strength,
                    <double> self.l2_strength,
                    <int> self.max_iter,
                    <double> self.tol,
                    <double> delta,
                    <int> self.linesearch_max_iter,
                    <int> self.lbfgs_memory,
                    <int> self.verbose,
                    <double*> coef_ptr,
                    <double*> &objective64,
                    <int*> &num_iters,
                    <int> self.loss_type,
                    <double*> sample_weight_ptr)

            self.objective = objective64

        self.num_iters = num_iters

        self._calc_intercept()

        self.handle.sync()

        del X_m
        del y_m

        return self

    @cuml.internals.api_base_return_array_skipall
    def _decision_function(self, X, convert_dtype=False) -> CumlArray:
        """
        Gives confidence score for X

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.

        Returns
        ----------
        y: array-like (device)
            Dense matrix (floats or doubles) of shape (n_samples, n_classes)
        """
        sparse_input = is_sparse(X)
        # Handle sparse inputs
        if sparse_input:
            X_m = SparseCumlArray(
                X,
                convert_to_dtype=(self.dtype if convert_dtype else None),
                convert_index=np.int32
            )
            n_rows, n_cols = X_m.shape
            self.dtype = X_m.dtype

        # Handle dense inputs
        else:
            X_m, n_rows, n_cols, self.dtype = input_to_cuml_array(
                X, check_dtype=self.dtype,
                convert_to_dtype=(self.dtype if convert_dtype else None),
                check_cols=self.n_cols,
                order='K'
            )

        scores = CumlArray.zeros(shape=(self._num_classes_dim, n_rows),
                                 dtype=self.dtype, order='F')

        cdef uintptr_t coef_ptr = self._coef_.ptr
        cdef uintptr_t scores_ptr = scores.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            if sparse_input:
                qnDecisionFunctionSparse(
                    handle_[0],
                    <float*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <float*> coef_ptr,
                    <int> self.loss_type,
                    <float*> scores_ptr)
            else:
                qnDecisionFunction(
                    handle_[0],
                    <float*><uintptr_t> X_m.ptr,
                    <bool> __is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <float*> coef_ptr,
                    <int> self.loss_type,
                    <float*> scores_ptr)

        else:
            if sparse_input:
                qnDecisionFunctionSparse(
                    handle_[0],
                    <double*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <double*> coef_ptr,
                    <int> self.loss_type,
                    <double*> scores_ptr)
            else:
                qnDecisionFunction(
                    handle_[0],
                    <double*><uintptr_t> X_m.ptr,
                    <bool> __is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <double*> coef_ptr,
                    <int> self.loss_type,
                    <double*> scores_ptr)

        self._calc_intercept()

        self.handle.sync()

        del X_m

        return scores

    @generate_docstring(
        X='dense_sparse',
        return_values={
            'name': 'preds',
            'type': 'dense',
            'description': 'Predicted values',
            'shape': '(n_samples, 1)'
        })
    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def predict(self, X, convert_dtype=False) -> CumlArray:
        """
        Predicts the y for X.

        """
        sparse_input = is_sparse(X)
        # Handle sparse inputs
        if sparse_input:
            X_m = SparseCumlArray(
                X,
                convert_to_dtype=(self.dtype if convert_dtype else None),
                convert_index=np.int32
            )
            n_rows, n_cols = X_m.shape
            self.dtype = X_m.dtype

        # Handle dense inputs
        else:
            X_m, n_rows, n_cols, self.dtype = input_to_cuml_array(
                X, check_dtype=self.dtype,
                convert_to_dtype=(self.dtype if convert_dtype else None),
                check_cols=self.n_cols,
                order='K'
            )

        preds = CumlArray.zeros(shape=n_rows, dtype=self.dtype,
                                index=X_m.index)
        cdef uintptr_t coef_ptr = self._coef_.ptr
        cdef uintptr_t pred_ptr = preds.ptr

        # temporary fix for dask-sql empty partitions
        if(n_rows == 0):
            return preds

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            if sparse_input:
                qnPredictSparse(
                    handle_[0],
                    <float*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <float*> coef_ptr,
                    <int> self.loss_type,
                    <float*> pred_ptr)
            else:
                qnPredict(
                    handle_[0],
                    <float*><uintptr_t> X_m.ptr,
                    <bool> __is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <float*> coef_ptr,
                    <int> self.loss_type,
                    <float*> pred_ptr)

        else:
            if sparse_input:
                qnPredictSparse(
                    handle_[0],
                    <double*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <double*> coef_ptr,
                    <int> self.loss_type,
                    <double*> pred_ptr)
            else:
                qnPredict(
                    handle_[0],
                    <double*><uintptr_t> X_m.ptr,
                    <bool> __is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> self._num_classes,
                    <bool> self.fit_intercept,
                    <double*> coef_ptr,
                    <int> self.loss_type,
                    <double*> pred_ptr)

        self._calc_intercept()

        self.handle.sync()

        del X_m

        return preds

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _calc_intercept(self):
        """
        If `fit_intercept == True`, then the last row of `coef_` contains
        `intercept_`. This should be called after every function that sets
        `coef_`
        """

        if (self.fit_intercept):
            self.intercept_ = self._coef_[-1]
        else:
            self.intercept_ = CumlArray.zeros(shape=1)

    def get_param_names(self):
        return super().get_param_names() + \
            ['loss', 'fit_intercept', 'l1_strength', 'l2_strength',
                'max_iter', 'tol', 'linesearch_max_iter', 'lbfgs_memory',
                'warm_start', 'delta']


def __is_col_major(X):
    return getattr(X, "order", "F").upper() == "F"
