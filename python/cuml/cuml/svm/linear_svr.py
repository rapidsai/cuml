# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

import cuml.svm.linear
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import RegressorMixin
from cuml.linear_model.base import LinearPredictMixin

__all__ = ["LinearSVR"]


class LinearSVR(Base, InteropMixin, LinearPredictMixin, RegressorMixin):
    """
    Linear Support Vector Regression.

    Similar to SVR with parameter kernel='linear', but implemented using a
    linear solver. This enables flexibility in penalties and loss functions,
    and can scale better for larger problems.

    Parameters
    ----------
    epsilon : float, default=0.0
        Epsilon parameter in the epsilon-insensitive loss function.
    penalty : {'l1', 'l2'}, default = 'l1'
        The norm used in the penalization.
    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'}, \
        default='epsilon_insensitive'
        The loss function.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    fit_intercept : bool, default=True
        Whether to fit the bias term. Set to False if you expect that the
        data is already centered.
    penalized_intercept : bool, default=False
        When true, the bias term is treated the same way as other features;
        i.e. it's penalized by the regularization term of the target function.
        Enabling this feature forces an extra copying the input data X.
    tol : float, default=1e-4
        Tolerance for the stopping criterion.
    max_iter : int, default=1000
        Maximum number of iterations for the underlying solver.
    linesearch_max_iter : int, default=100
        Maximum number of linesearch (inner loop) iterations for
        the underlying (QN) solver.
    lbfgs_memory : int, default=5
        Number of vectors approximating the hessian for the underlying QN
        solver (l-bfgs).
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
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    coef_ : array, shape (1, n_features)
        Weights assigned to the features (coefficients in the primal problem).
    intercept_ : array or float, shape (1,)
        The constant factor in the decision function. If
        ``fit_intercept=False`` this is instead a float with value 0.0.

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVR>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's LinearSVR
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.svm import LinearSVR
    >>> X = cp.array([[1], [2], [3], [4], [5]], dtype=cp.float32)
    >>> y = cp.array([1.1, 4, 5, 3.9, 8.], dtype=cp.float32)
    >>> reg = LinearSVR(epsilon=0.1, C=10).fit(X, y)
    >>> print("Predicted values:", reg.predict(X)) # doctest: +SKIP
    Predicted values: [1.8993504 3.3995128 4.899675  6.399837  7.899999]
    """

    coef_ = CumlArrayDescriptor(order="F")
    intercept_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.svm.LinearSVR"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "epsilon",
            "penalty",
            "loss",
            "C",
            "fit_intercept",
            "penalized_intercept",
            "tol",
            "max_iter",
            "linesearch_max_iter",
            "lbfgs_memory",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.intercept_scaling != 1:
            raise UnsupportedOnGPU(
                f"`intercept_scaling={model.intercept_scaling}` is not supported"
            )

        # Infer the penalty from the loss
        penalty = "l1" if model.loss == "epsilon_insensitive" else "l2"

        return {
            "epsilon": model.epsilon,
            "penalty": penalty,
            "loss": model.loss,
            "C": model.C,
            "fit_intercept": model.fit_intercept,
            "tol": model.tol,
            "max_iter": model.max_iter,
        }

    def _params_to_cpu(self):
        return {
            "epsilon": self.epsilon,
            "loss": self.loss,
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def _attrs_from_cpu(self, model):
        return {
            "coef_": to_gpu(model.coef_, order="F", dtype=np.float64),
            "intercept_": to_gpu(
                model.intercept_, order="F", dtype=np.float64
            ),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "coef_": to_cpu(self.coef_, order="C", dtype=np.float64),
            "intercept_": to_cpu(self.intercept_, order="C", dtype=np.float64),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        epsilon=0.0,
        penalty="l1",
        loss="epsilon_insensitive",
        C=1.0,
        fit_intercept=True,
        penalized_intercept=False,
        tol=1e-4,
        max_iter=1000,
        linesearch_max_iter=100,
        lbfgs_memory=5,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        self.epsilon = epsilon
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalized_intercept = penalized_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory

    @generate_docstring()
    def fit(
        self, X, y, sample_weight=None, *, convert_dtype=True
    ) -> "LinearSVR":
        """Fit the model according to the given training data."""
        X = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        ).array

        y = input_to_cuml_array(
            y,
            check_dtype=X.dtype,
            convert_to_dtype=(X.dtype if convert_dtype else None),
            check_rows=X.shape[0],
            check_cols=1,
        ).array

        if sample_weight is not None:
            sample_weight = input_to_cuml_array(
                sample_weight,
                check_dtype=X.dtype,
                convert_to_dtype=(X.dtype if convert_dtype else None),
                check_rows=X.shape[0],
                check_cols=1,
            ).array

        coef, intercept, _ = cuml.svm.linear.fit(
            self.handle,
            X,
            y,
            sample_weight=sample_weight,
            loss=self.loss,
            penalty=self.penalty,
            fit_intercept=self.fit_intercept,
            penalized_intercept=self.penalized_intercept,
            max_iter=self.max_iter,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            C=self.C,
            tol=self.tol,
            epsilon=self.epsilon,
            verbose=self.verbose,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        return self
