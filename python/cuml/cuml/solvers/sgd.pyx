# Copyright (c) 2018-2025, NVIDIA CORPORATION.
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
import cupy as cp
import numpy as np

from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.mixins import FMajorInputTagMixin

from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/solvers/solver.hpp" namespace "ML::Solver" nogil:

    cdef void sgdFit(handle_t& handle,
                     float *input,
                     int n_rows,
                     int n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     int batch_size,
                     int epochs,
                     int lr_type,
                     float eta0,
                     float power_t,
                     int loss,
                     int penalty,
                     float alpha,
                     float l1_ratio,
                     bool shuffle,
                     float tol,
                     int n_iter_no_change) except +

    cdef void sgdFit(handle_t& handle,
                     double *input,
                     int n_rows,
                     int n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     int batch_size,
                     int epochs,
                     int lr_type,
                     double eta0,
                     double power_t,
                     int loss,
                     int penalty,
                     double alpha,
                     double l1_ratio,
                     bool shuffle,
                     double tol,
                     int n_iter_no_change) except +

    cdef void sgdPredict(handle_t& handle,
                         const float *input,
                         int n_rows,
                         int n_cols,
                         const float *coef,
                         float intercept,
                         float *preds,
                         int loss) except +

    cdef void sgdPredict(handle_t& handle,
                         const double *input,
                         int n_rows,
                         int n_cols,
                         const double *coef,
                         double intercept,
                         double *preds,
                         int loss) except +

    cdef void sgdPredictBinaryClass(handle_t& handle,
                                    const float *input,
                                    int n_rows,
                                    int n_cols,
                                    const float *coef,
                                    float intercept,
                                    float *preds,
                                    int loss) except +

    cdef void sgdPredictBinaryClass(handle_t& handle,
                                    const double *input,
                                    int n_rows,
                                    int n_cols,
                                    const double *coef,
                                    double intercept,
                                    double *preds,
                                    int loss) except +


class SGD(Base,
          FMajorInputTagMixin):
    """
    Stochastic Gradient Descent is a very common machine learning algorithm
    where one optimizes some cost function via gradient steps. This makes SGD
    very attractive for large problems when the exact solution is hard or even
    impossible to find.

    cuML's SGD algorithm accepts a numpy matrix or a cuDF DataFrame as the
    input dataset. The SGD algorithm currently works with linear regression,
    ridge regression and SVM models.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import cudf
        >>> from cuml.solvers import SGD as cumlSGD
        >>> X = cudf.DataFrame()
        >>> X['col1'] = np.array([1,1,2,2], dtype=np.float32)
        >>> X['col2'] = np.array([1,2,2,3], dtype=np.float32)
        >>> y = cudf.Series(np.array([1, 1, 2, 2], dtype=np.float32))
        >>> pred_data = cudf.DataFrame()
        >>> pred_data['col1'] = np.asarray([3, 2], dtype=np.float32)
        >>> pred_data['col2'] = np.asarray([5, 5], dtype=np.float32)
        >>> cu_sgd = cumlSGD(learning_rate='constant', eta0=0.005, epochs=2000,
        ...                  fit_intercept=True, batch_size=2,
        ...                  tol=0.0, penalty=None, loss='squared_loss')
        >>> cu_sgd.fit(X, y)
        SGD()
        >>> cu_pred = cu_sgd.predict(pred_data).to_numpy()
        >>> print(" cuML intercept : ", cu_sgd.intercept_) # doctest: +SKIP
        cuML intercept :  0.00418...
        >>> print(" cuML coef : ", cu_sgd.coef_) # doctest: +SKIP
        cuML coef :  0      0.9841...
        1      0.0097...
        dtype: float32
        >>> print("cuML predictions : ", cu_pred) # doctest: +SKIP
        cuML predictions :  [3.0055...  2.0214...]

    Parameters
    ----------
    loss : 'hinge', 'log', 'squared_loss' (default = 'squared_loss')
        'hinge' uses linear SVM
        'log' uses logistic regression
        'squared_loss' uses linear regression
    penalty : {'l1', 'l2', 'elasticnet', None} (default = None)
        The penalty (aka regularization term) to apply.

        - 'l1': L1 norm (Lasso) regularization
        - 'l2': L2 norm (Ridge) regularization
        - 'elasticnet': Elastic Net regularization, a weighted average of L1 and L2
        - None: no penalty is added (the default)

    alpha : float (default = 0.0001)
        The constant value which decides the degree of regularization
    fit_intercept : boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    epochs : int (default = 1000)
        The number of times the model should iterate through the entire dataset
        during training (default = 1000)
    tol : float (default = 1e-3)
        The training process will stop if current_loss > previous_loss - tol
    shuffle : boolean (default = True)
        True, shuffles the training data after each epoch
        False, does not shuffle the training data after each epoch
    eta0 : float (default = 0.001)
        Initial learning rate
    power_t : float (default = 0.5)
        The exponent used for calculating the invscaling learning rate
    batch_size : int (default=32)
        The number of samples to use for each batch.
    learning_rate : 'optimal', 'constant', 'invscaling', \
                    'adaptive' (default = 'constant')
        Optimal option supported in the next version
        constant keeps the learning rate constant
        adaptive changes the learning rate if the training loss or the
        validation accuracy does not improve for n_iter_no_change epochs.
        The old learning rate is generally divide by 5
    n_iter_no_change : int (default = 5)
        The number of epochs to train without any improvement in the model
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    """

    coef_ = CumlArrayDescriptor()
    classes_ = CumlArrayDescriptor()

    def __init__(self, *, loss='squared_loss', penalty=None, alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, epochs=1000, tol=1e-3,
                 shuffle=True, learning_rate='constant', eta0=0.001,
                 power_t=0.5, batch_size=32, n_iter_no_change=5, handle=None,
                 output_type=None, verbose=False):

        if loss in ['hinge', 'log', 'squared_loss']:
            self.loss = loss
        else:
            raise ValueError(f"loss {loss!r} is not supported")

        if penalty in [None, 'l1', 'l2', 'elasticnet']:
            self.penalty = penalty
        else:
            raise ValueError(f"penalty {penalty!r} is not supported")

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.shuffle = shuffle
        self.eta0 = eta0
        self.power_t = power_t

        if learning_rate in ['optimal', 'constant', 'invscaling', 'adaptive']:
            self.learning_rate = learning_rate

            if learning_rate in ["constant", "invscaling", "adaptive"]:
                if self.eta0 <= 0.0:
                    raise ValueError("eta0 must be > 0")

            if learning_rate == 'optimal':
                self.lr_type = 0

                raise TypeError("This option will be supported in the future")

                # TODO: uncomment this when optimal learning rate is supported
                # if self.alpha == 0:
                #     raise ValueError("alpha must be > 0 since "
                #                      "learning_rate is 'optimal'. alpha is "
                #                      "used to compute the optimal learning "
                #                      " rate.")

            elif learning_rate == 'constant':
                self.lr_type = 1
                self.lr = eta0
            elif learning_rate == 'invscaling':
                self.lr_type = 2
            elif learning_rate == 'adaptive':
                self.lr_type = 3
        else:
            msg = "learning rate {!r} is not supported"
            raise TypeError(msg.format(learning_rate))

        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change
        self.intercept_value = 0.0

    def _check_alpha(self, alpha):
        for el in alpha:
            if el <= 0.0:
                msg = "alpha values have to be positive"
                raise TypeError(msg.format(alpha))

    def _get_loss_int(self):
        return {
            'squared_loss': 0,
            'log': 1,
            'hinge': 2,
        }[self.loss]

    def _get_penalty_int(self):
        return {
            None: 0,
            'l1': 1,
            'l2': 2,
            'elasticnet': 3
        }[self.penalty]

    @generate_docstring()
    @cuml.internals.api_base_return_any(set_output_dtype=True)
    def fit(self, X, y, *, convert_dtype=True) -> "SGD":
        """
        Fit the model with X and y.

        """
        X_m, n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else None),
                                check_dtype=[np.float32, np.float64])

        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows, check_cols=1)

        _estimator_type = getattr(self, '_estimator_type', None)
        if _estimator_type == "classifier":
            self.classes_ = cp.unique(y_m)

        cdef uintptr_t _X_ptr = X_m.ptr
        cdef uintptr_t _y_ptr = y_m.ptr

        self.n_alpha = 1

        self.coef_ = CumlArray.zeros(self.n_cols,
                                     dtype=self.dtype)
        cdef uintptr_t _coef_ptr = self.coef_.ptr

        cdef float _c_intercept_f32
        cdef double _c_intercept_f64

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            sgdFit(handle_[0],
                   <float*>_X_ptr,
                   <int>n_rows,
                   <int>self.n_cols,
                   <float*>_y_ptr,
                   <float*>_coef_ptr,
                   <float*>&_c_intercept_f32,
                   <bool>self.fit_intercept,
                   <int>self.batch_size,
                   <int>self.epochs,
                   <int>self.lr_type,
                   <float>self.eta0,
                   <float>self.power_t,
                   <int>self._get_loss_int(),
                   <int>self._get_penalty_int(),
                   <float>self.alpha,
                   <float>self.l1_ratio,
                   <bool>self.shuffle,
                   <float>self.tol,
                   <int>self.n_iter_no_change)

            self.intercept_ = _c_intercept_f32
        else:
            sgdFit(handle_[0],
                   <double*>_X_ptr,
                   <int>n_rows,
                   <int>self.n_cols,
                   <double*>_y_ptr,
                   <double*>_coef_ptr,
                   <double*>&_c_intercept_f64,
                   <bool>self.fit_intercept,
                   <int>self.batch_size,
                   <int>self.epochs,
                   <int>self.lr_type,
                   <double>self.eta0,
                   <double>self.power_t,
                   <int>self._get_loss_int(),
                   <int>self._get_penalty_int(),
                   <double>self.alpha,
                   <double>self.l1_ratio,
                   <bool>self.shuffle,
                   <double>self.tol,
                   <int>self.n_iter_no_change)

            self.intercept_ = _c_intercept_f64

        self.handle.sync()

        del X_m
        del y_m

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        X_m, _n_rows, _n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t _X_ptr = X_m.ptr

        cdef uintptr_t _coef_ptr = self.coef_.ptr
        preds = CumlArray.zeros(_n_rows, dtype=self.dtype, index=X_m.index)
        cdef uintptr_t _preds_ptr = preds.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            sgdPredict(handle_[0],
                       <float*>_X_ptr,
                       <int>_n_rows,
                       <int>_n_cols,
                       <float*>_coef_ptr,
                       <float>self.intercept_,
                       <float*>_preds_ptr,
                       <int>self._get_loss_int())
        else:
            sgdPredict(handle_[0],
                       <double*>_X_ptr,
                       <int>_n_rows,
                       <int>_n_cols,
                       <double*>_coef_ptr,
                       <double>self.intercept_,
                       <double*>_preds_ptr,
                       <int>self._get_loss_int())

        self.handle.sync()

        del X_m

        return preds

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def predictClass(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        X_m, _n_rows, _n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t _X_ptr = X_m.ptr
        cdef uintptr_t _coef_ptr = self.coef_.ptr
        preds = CumlArray.zeros(_n_rows, dtype=dtype, index=X_m.index)
        cdef uintptr_t _preds_ptr = preds.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            sgdPredictBinaryClass(handle_[0],
                                  <float*>_X_ptr,
                                  <int>_n_rows,
                                  <int>_n_cols,
                                  <float*>_coef_ptr,
                                  <float>self.intercept_,
                                  <float*>_preds_ptr,
                                  <int>self._get_loss_int())
        else:
            sgdPredictBinaryClass(handle_[0],
                                  <double*>_X_ptr,
                                  <int>_n_rows,
                                  <int>_n_cols,
                                  <double*>_coef_ptr,
                                  <double>self.intercept_,
                                  <double*>_preds_ptr,
                                  <int>self._get_loss_int())

        self.handle.sync()

        del X_m

        return preds

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "loss",
            "penalty",
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "epochs",
            "tol",
            "shuffle",
            "learning_rate",
            "eta0",
            "power_t",
            "batch_size",
            "n_iter_no_change",
        ]
