# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from pylibraft.common.handle import Handle

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.mixins import FMajorInputTagMixin

from libc.stdint cimport uintptr_t
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


_LEARNING_RATES = {
    "constant": 1,
    "invscaling": 2,
    "adaptive": 3,
}

_LOSSES = {
    "squared_loss": 0,
    "log": 1,
    "hinge": 2,
}

_PENALTIES = {
    None: 0,
    "l1": 1,
    "l2": 2,
    "elasticnet": 3
}


def fit_sgd(
    X,
    y,
    *,
    convert_dtype=True,
    loss="squared_loss",
    penalty=None,
    double alpha=0.0001,
    double l1_ratio=0.15,
    bool fit_intercept=True,
    int epochs=1000,
    double tol=1e-3,
    bool shuffle=True,
    learning_rate="constant",
    double eta0=0.001,
    double power_t=0.5,
    int batch_size=32,
    int n_iter_no_change=5,
    handle=None,
):
    """Fit a linear model using stochastic gradient descent.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The training data.
    y : array-like, shape=(n_samples,)
        The target values.
    convert_to_dtype : bool, default=True
        When set to True, will convert array inputs to be of the proper dtypes.
    **kwargs
        Remaining keyword arguments match the hyperparameters
        to ``SGD``, see the ``SGD`` docs for more information.

    Returns
    -------
    coef : CumlArray, shape=(n_features,)
        The fit coefficients
    intercept : float
        The fit intercept, or 0 if `fit_intercept=False`
    """
    # Validate parameters
    if eta0 <= 0.0:
        raise ValueError(f"Expected eta0 > 0, got {eta0}")

    if alpha <= 0.0:
        raise ValueError(f"Expected alpha > 0, got {alpha}")

    cdef int lr_code
    if (lr_code := _LEARNING_RATES.get(learning_rate, -1)) < 0:
        raise ValueError(
            f"Expected `learning_rate` to be one of {list(_LEARNING_RATES)},"
            f" got {learning_rate!r}"
        )

    cdef int penalty_code
    if (penalty_code := _PENALTIES.get(penalty, -1)) < 0:
        raise ValueError(
            f"Expected `penalty` to be one of {list(_PENALTIES)}, got {penalty!r}"
        )

    cdef int loss_code
    if (loss_code := _LOSSES.get(loss, -1)) < 0:
        raise ValueError(
            f"Expected `loss` to be one of {list(_LOSSES)}, got {loss!r}"
        )

    # Validate X and y
    cdef int n_rows, n_cols
    X, n_rows, n_cols, _ = input_to_cuml_array(
        X,
        convert_to_dtype=(np.float32 if convert_dtype else None),
        check_dtype=[np.float32, np.float64],
    )

    y = input_to_cuml_array(
        y,
        check_dtype=X.dtype,
        convert_to_dtype=(X.dtype if convert_dtype else None),
        check_rows=X.shape[0],
        check_cols=1,
    ).array

    # Allocate outputs
    coef = CumlArray.zeros(n_cols, dtype=X.dtype)

    # Perform fit
    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef uintptr_t X_ptr = X.ptr
    cdef uintptr_t y_ptr = y.ptr
    cdef uintptr_t coef_ptr = coef.ptr
    cdef float intercept_f32
    cdef double intercept_f64
    cdef bool use_f32 = X.dtype == np.float32

    with nogil:
        if use_f32:
            sgdFit(
                handle_[0],
                <float*>X_ptr,
                n_rows,
                n_cols,
                <float*>y_ptr,
                <float*>coef_ptr,
                &intercept_f32,
                fit_intercept,
                batch_size,
                epochs,
                lr_code,
                eta0,
                power_t,
                loss_code,
                penalty_code,
                alpha,
                l1_ratio,
                shuffle,
                tol,
                n_iter_no_change
            )
        else:
            sgdFit(
                handle_[0],
                <double*>X_ptr,
                n_rows,
                n_cols,
                <double*>y_ptr,
                <double*>coef_ptr,
                &intercept_f64,
                fit_intercept,
                batch_size,
                epochs,
                lr_code,
                eta0,
                power_t,
                loss_code,
                penalty_code,
                alpha,
                l1_ratio,
                shuffle,
                tol,
                n_iter_no_change
            )
    handle.sync()

    return coef, (intercept_f32 if use_f32 else intercept_f64)


class SGD(Base, FMajorInputTagMixin):
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
    learning_rate : {'constant', 'invscaling', 'adaptive'} (default = 'constant')
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

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
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

    def __init__(
        self,
        *,
        loss="squared_loss",
        penalty=None,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        epochs=1000,
        tol=1e-3,
        shuffle=True,
        learning_rate="constant",
        eta0=0.001,
        power_t=0.5,
        batch_size=32,
        n_iter_no_change=5,
        handle=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change

    @generate_docstring()
    def fit(self, X, y, *, convert_dtype=True) -> "SGD":
        """
        Fit the model with X and y.

        """
        coef, intercept = fit_sgd(
            X,
            y,
            convert_dtype=convert_dtype,
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            epochs=self.epochs,
            tol=self.tol,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            batch_size=self.batch_size,
            n_iter_no_change=self.n_iter_no_change,
            handle=self.handle,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples,)"
        }
    )
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        cdef int n_rows, n_cols
        X, n_rows, n_cols, _ = input_to_cuml_array(
            X,
            check_dtype=self.coef_.dtype,
            convert_to_dtype=(self.coef_.dtype if convert_dtype else None),
            check_cols=self.coef_.shape[0],
        )

        preds = CumlArray.zeros(n_rows, dtype=self.coef_.dtype, index=X.index)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef int loss_code = _LOSSES[self.loss]
        cdef bool use_f32 = self.coef_.dtype == np.float32
        cdef uintptr_t preds_ptr = preds.ptr
        cdef uintptr_t X_ptr = X.ptr
        cdef uintptr_t coef_ptr = self.coef_.ptr
        cdef double intercept = self.intercept_

        with nogil:
            if use_f32:
                sgdPredict(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>coef_ptr,
                    intercept,
                    <float*>preds_ptr,
                    loss_code,
                )
            else:
                sgdPredict(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>coef_ptr,
                    intercept,
                    <double*>preds_ptr,
                    loss_code,
                )
        self.handle.sync()

        return preds

    def predictClass(self, X, convert_dtype=True):
        """This method has been removed.

        Instead use ``sgd.predict() > 0.5`` for ``loss="hinge"`` and
        ``sgd.predict() > 0`` otherwise. For actual classifier support
        please use ``MBSGDClassifier`` instead.
        """
        raise NotImplementedError(
            "This method was removed in 25.12 as a breaking change.\n\n"
            "Please use ``sgd.predict() > 0.5`` for ``loss='hinge'`` and "
            "``sgd.predict() > 0`` otherwise."
        )
