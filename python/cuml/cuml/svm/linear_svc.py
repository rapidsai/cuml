# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

import cuml.svm.linear
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.classification import (
    check_classification_targets,
    process_class_weight,
)
from cuml.common.doc_utils import generate_docstring
from cuml.common.exceptions import NotFittedError
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import ClassifierMixin
from cuml.linear_model.base import LinearClassifierMixin

__all__ = ("LinearSVC",)


class LinearSVC(Base, InteropMixin, LinearClassifierMixin, ClassifierMixin):
    """
    Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented using a
    linear solver. This enables flexibility in penalties and loss functions,
    and can scale better for larger problems.

    Parameters
    ----------
    penalty : {'l1', 'l2'}, default = 'l2'
        The norm used in the penalization.
    loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
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
    class_weight : dict or string, default=None
        Weights to modify the parameter C for class i to ``class_weight[i]*C``.
        The string 'balanced' is also accepted, in which case
        ``class_weight[i] = n_samples / (n_classes * n_samples_of_class[i])``
    probability: bool, default=False
        Set to True to enable probability estimate methods (``predict_proba``,
        ``predict_log_proba``).
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
    multi_class : {'ovr'}, default='ovr'
        Multiclass classification strategy. Currently only 'ovr' is supported.
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
    coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes, n_features)
        Weights assigned to the features (coefficients in the primal problem).
    intercept_ : array or float, shape (1,) if n_classes == 2 else (n_classes,)
        The constant factor in the decision function. If
        ``fit_intercept=False`` this is instead a float with value 0.0.
    classes_ : array, shape (n_classes,)
        The unique class labels
    n_iter_ : int
        The maximum number of iterations run across all classes during the fit.
    prob_scale_ : array or None, shape (`n_classes_`, 2)
        The probability calibration constants if ``probability=True``,
        otherwise ``None``.

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVC>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's LinearSVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.svm import LinearSVC
    >>> X = cp.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
    ...              dtype=cp.float32);
    >>> y = cp.array([0, 0, 1, 0, 1, 1], dtype=cp.float32)
    >>> clf = LinearSVC(penalty='l1', C=1).fit(X, y)
    >>> print("Predicted labels:", clf.predict(X))  # doctest: +SKIP
    Predicted labels: [0 0 1 0 1 1]
    """

    coef_ = CumlArrayDescriptor(order="F")
    intercept_ = CumlArrayDescriptor(order="F")
    classes_ = CumlArrayDescriptor(order="F")
    prob_scale_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.svm.LinearSVC"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "penalty",
            "loss",
            "C",
            "fit_intercept",
            "penalized_intercept",
            "class_weight",
            "probability",
            "tol",
            "max_iter",
            "linesearch_max_iter",
            "lbfgs_memory",
            "multi_class",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.intercept_scaling != 1:
            raise UnsupportedOnGPU(
                f"`intercept_scaling={model.intercept_scaling}` is not supported"
            )
        if model.multi_class != "ovr":
            raise UnsupportedOnGPU(
                f"`multi_class={model.multi_class}` is not supported"
            )

        return {
            "penalty": model.penalty,
            "loss": model.loss,
            "C": model.C,
            "fit_intercept": model.fit_intercept,
            "class_weight": model.class_weight,
            "tol": model.tol,
            "max_iter": model.max_iter,
            "multi_class": model.multi_class,
        }

    def _params_to_cpu(self):
        return {
            "penalty": self.penalty,
            "loss": self.loss,
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "class_weight": self.class_weight,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "multi_class": self.multi_class,
        }

    def _attrs_from_cpu(self, model):
        return {
            "coef_": to_gpu(model.coef_, order="F", dtype=np.float64),
            "intercept_": to_gpu(
                model.intercept_, order="F", dtype=np.float64
            ),
            "classes_": to_gpu(model.classes_, order="F"),
            "prob_scale_": None,
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "coef_": to_cpu(self.coef_, order="C", dtype=np.float64),
            "intercept_": to_cpu(self.intercept_, order="C", dtype=np.float64),
            "classes_": to_cpu(self.classes_, order="C"),
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        penalty="l2",
        loss="squared_hinge",
        C=1.0,
        fit_intercept=True,
        penalized_intercept=False,
        class_weight=None,
        probability=False,
        tol=1e-4,
        max_iter=1000,
        linesearch_max_iter=100,
        lbfgs_memory=5,
        multi_class="ovr",
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalized_intercept = penalized_intercept
        self.class_weight = class_weight
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.multi_class = multi_class

    @generate_docstring()
    def fit(
        self, X, y, sample_weight=None, *, convert_dtype=True
    ) -> "LinearSVC":
        """Fit the model according to the given training data."""
        X = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        ).array

        y = input_to_cupy_array(y, check_rows=X.shape[0], check_cols=1).array
        check_classification_targets(y)
        classes, y = cp.unique(y, return_inverse=True)

        _, sample_weight = process_class_weight(
            classes,
            y,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
            float64=(X.dtype == np.float64),
        )

        coef, intercept, n_iter, prob_scale = cuml.svm.linear.fit(
            self.handle,
            X,
            CumlArray(data=y.astype(X.dtype, copy=False)),
            sample_weight=sample_weight,
            classes=CumlArray(data=classes.astype(X.dtype, copy=False)),
            probability=self.probability,
            loss=self.loss,
            penalty=self.penalty,
            fit_intercept=self.fit_intercept,
            penalized_intercept=self.penalized_intercept,
            max_iter=self.max_iter,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            C=self.C,
            tol=self.tol,
            epsilon=0.0,
            verbose=self.verbose,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        self.classes_ = CumlArray(data=classes)
        self.n_iter_ = n_iter
        self.prob_scale_ = prob_scale
        return self

    @generate_docstring(
        return_values={
            "name": "y_pred",
            "type": "dense",
            "description": "Predicted class labels.",
            "shape": "(n_samples,)",
        },
    )
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """Predict class labels for samples in X."""
        classes = self.classes_.to_output("cupy")

        if self.probability:
            probs = self.predict_proba(X, convert_dtype=convert_dtype)
            inds = probs.to_output("cupy").argmax(axis=1)
        else:
            scores = self.decision_function(X, convert_dtype=convert_dtype)
            if scores.ndim == 1:
                inds = scores.to_output("cupy") >= 0
            else:
                inds = scores.to_output("cupy").argmax(axis=1)
        return CumlArray(data=classes.take(inds))

    @generate_docstring(
        return_values={
            "name": "probs",
            "type": "dense",
            "description": "Probabilities per class for each sample.",
            "shape": "(n_samples, n_classes)",
        },
    )
    def predict_proba(self, X, *, convert_dtype=True) -> CumlArray:
        """Compute probabilities of possible outcomes for samples in X.

        The model must have been fit with ``probability=True`` for this method
        to be available.
        """
        if self.prob_scale_ is None:
            raise NotFittedError(
                "This classifier is not fitted to predict "
                "probabilities. Fit a new classifier with "
                "probability=True to enable predict_proba."
            )

        scores = self.decision_function(X, convert_dtype=convert_dtype)
        scores = input_to_cuml_array(
            scores,
            check_dtype=self.coef_.dtype,
            order="C",
        ).array
        return cuml.svm.linear.compute_probabilities(
            self.handle, scores, self.prob_scale_
        )

    @generate_docstring(
        return_values={
            "name": "probs",
            "type": "dense",
            "description": "Log probabilities per class for each sample.",
            "shape": "(n_samples, n_classes)",
        },
    )
    def predict_log_proba(self, X, *, convert_dtype=True) -> CumlArray:
        """Compute log probabilities of possible outcomes for samples in X.

        The model must have been fit with ``probability=True`` for this method
        to be available.
        """
        probs = self.predict_proba(X, convert_dtype=convert_dtype).to_output(
            "cupy"
        )
        cp.log(probs, out=probs)
        return CumlArray(data=probs)
