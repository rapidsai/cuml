#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numpy as np

import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import ClassifierMixin, SparseInputTagMixin
from cuml.internals.output_utils import cudf_to_pandas
from cuml.linear_model.base import LinearClassifierMixin
from cuml.preprocessing import LabelEncoder
from cuml.solvers.qn import fit_qn


class LogisticRegression(
    Base,
    InteropMixin,
    LinearClassifierMixin,
    ClassifierMixin,
    SparseInputTagMixin,
):
    """Logistic Regression classifier.

    LogisticRegression is a linear model that is used to model probability of
    occurrence of certain events, for example probability of success or fail of
    an event.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Specifies the penalty term to use. `'l1'` and `'l2'` will use an L1 or
        L2 penalty, respectively. `'elasticnet'` will use both an L1 and L2
        penalty. `None` will not use a penalty.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
        The exact stopping conditions depend on the chosen solver.
        Check the solver's documentation for more details:

          * :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a bias or intercept) should be added to
        the decision function. Note that, just like in Scikit-learn, the bias
        will not be regularized.
    class_weight : dict or 'balanced', default=None
        By default all classes have a weight one. However, a dictionary
        can be provided with weights associated with classes
        in the form ``{class_label: weight}``. The "balanced" mode
        uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``. Note that
        these weights will be multiplied with sample_weight
        (passed through the fit method) if sample_weight is specified.
    max_iter : int, default=1000
        Maximum number of iterations taken for the solvers to converge.
    linesearch_max_iter : int, default=50
        Max number of linesearch iterations per outer iteration used in the
        lbfgs and owl QN solvers.
    l1_ratio : float or None, default=None
        The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`
    solver : {'qn'}, default='qn'
        Algorithm to use in the optimization problem. Currently only `qn` is
        supported, which automatically selects either L-BFGS or OWL-QN
        depending on the conditions of the l1 regularization described
        above.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
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

    Attributes
    ----------
    coef_: array, shape=(n_classes, n_features) or (n_classes, n_features)
        The estimated coefficients for the logistic regression model.
    intercept_: array, shape=(1,) or (n_classes,)
        The independent term. If `fit_intercept` is False, will be 0.
    n_iter_: array, shape (1,)
        The number of iterations taken for the solvers to converge.

    Notes
    -----
    cuML's LogisticRegression uses a different solver that the equivalent
    Scikit-learn, except when there is no penalty and `solver=lbfgs` is
    used in Scikit-learn. This can cause (smaller) differences in the
    coefficients and predictions of the model, similar to
    using different solvers in Scikit-learn.

    For additional information, see `Scikit-learn's LogisticRegression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

    Examples
    --------
    >>> import cuml
    >>> import cupy as cp
    >>> X = cp.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = cp.array([0, 0, 1, 1])
    >>> model = cuml.LogisticRegression().fit(X, y)
    >>> model.predict(X)
    array([0, 0, 1, 1])
    """

    _cpu_class_path = "sklearn.linear_model.LogisticRegression"

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "penalty",
            "tol",
            "C",
            "fit_intercept",
            "class_weight",
            "max_iter",
            "linesearch_max_iter",
            "l1_ratio",
            "solver",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        if model.intercept_scaling != 1:
            raise UnsupportedOnGPU(
                f"`intercept_scaling={model.intercept_scaling}` is not supported"
            )

        # `multi_class` was deprecated in sklearn 1.5 and will be removed in 1.8
        if getattr(model, "multi_class", "deprecated") not in (
            "deprecated",
            "auto",
        ):
            raise UnsupportedOnGPU("`multi_class` is not supported")

        return {
            "penalty": model.penalty,
            "tol": model.tol,
            "C": model.C,
            "fit_intercept": model.fit_intercept,
            "class_weight": model.class_weight,
            "max_iter": model.max_iter,
            "l1_ratio": model.l1_ratio,
            "solver": "qn",
        }

    def _params_to_cpu(self):
        return {
            "penalty": self.penalty,
            "tol": self.tol,
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "l1_ratio": self.l1_ratio,
            "solver": "lbfgs" if self.penalty in ("l2", None) else "saga",
        }

    def _attrs_from_cpu(self, model):
        return {
            "classes_": model.classes_,
            "intercept_": to_gpu(model.intercept_, order="F"),
            "coef_": to_gpu(model.coef_, order="F"),
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "classes_": self.classes_,
            "intercept_": to_cpu(self.intercept_),
            "coef_": to_cpu(self.coef_),
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        penalty="l2",
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        class_weight=None,
        max_iter=1000,
        linesearch_max_iter=50,
        l1_ratio=None,
        solver="qn",
        verbose=False,
        handle=None,
        output_type=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.l1_ratio = l1_ratio
        self.solver = solver

    def _get_l1_l2_strength(self):
        if self.solver != "qn":
            raise ValueError(
                f"Only solver='qn' is supported, got {self.solver!r}"
            )

        if self.penalty is None:
            l1_strength = 0.0
            l2_strength = 0.0
        elif self.penalty == "l1":
            l1_strength = 1.0 / self.C
            l2_strength = 0.0
        elif self.penalty == "l2":
            l1_strength = 0.0
            l2_strength = 1.0 / self.C
        elif self.penalty == "elasticnet":
            if self.l1_ratio is None:
                raise ValueError(
                    "l1_ratio must be specified when penalty is elasticnet"
                )
            if self.l1_ratio < 0.0 or self.l1_ratio > 1.0:
                raise ValueError(
                    f"Expected 0 <= l1_ratio <= 1, got {self.l1_ratio}"
                )
            strength = 1.0 / self.C
            l1_strength = self.l1_ratio * strength
            l2_strength = (1.0 - self.l1_ratio) * strength
        else:
            raise ValueError(f"penalty={self.penalty!r} is not supported")

        return l1_strength, l2_strength

    @generate_docstring(X="dense_sparse")
    @cuml.internals.api_base_return_any()
    def fit(
        self, X, y, sample_weight=None, *, convert_dtype=True
    ) -> "LogisticRegression":
        """
        Fit the model with X and y.
        """
        # LabelEncoder currently always returns cudf (ignoring output_type),
        # but we need to explicitly set `output_type` or we'll get an error
        # on init since `output_type` would default to `mirror`.
        enc = LabelEncoder(output_type="cudf")
        y_orig_dtype = getattr(y, "dtype", None)
        y = enc.fit_transform(y).to_cupy()
        if y_orig_dtype is None:
            y_orig_dtype = y.dtype
        n_samples = len(y)
        classes = enc.classes_.to_numpy()

        # TODO: LabelEncoder doesn't currently map dtypes the same way as it
        # does in scikit-learn. Until that's fixed we fix them up here.
        if y_orig_dtype.kind == "U":
            classes = classes.astype("U")
        elif y_orig_dtype == "float16":
            classes = classes.astype("float16")
        self.classes_ = classes
        n_classes = len(self.classes_)

        # TODO: unify class_weight handling across all classifiers
        if self.class_weight is not None:
            # Coerce sample_weight to same dtype as X, falling back to float32
            dtype = getattr(X, "dtype", cp.float32)

            if sample_weight is not None:
                sample_weight = input_to_cuml_array(
                    sample_weight,
                    convert_to_dtype=dtype,
                    check_rows=n_samples,
                    check_cols=1,
                ).array.to_output("cupy")
            else:
                sample_weight = cp.ones(y.shape, dtype=dtype)

            if self.class_weight == "balanced":
                counts = cp.asnumpy(cp.bincount(y))
                weights = n_samples / (n_classes * counts)
                class_weight = {i: weights[i] for i in range(n_classes)}
            else:
                keys = self.class_weight.keys()
                encoded_keys = enc.transform(cudf.Series(keys)).values_host
                class_weight = {
                    enc_key: self.class_weight[key]
                    for enc_key, key in zip(encoded_keys, keys)
                }

            for label, weight in class_weight.items():
                sample_weight[y == label] *= weight

        l1_strength, l2_strength = self._get_l1_l2_strength()

        coef, intercept, n_iter, _ = fit_qn(
            X,
            y,
            sample_weight=sample_weight,
            convert_dtype=convert_dtype,
            n_classes=n_classes,
            loss=("softmax" if n_classes > 2 else "sigmoid"),
            fit_intercept=self.fit_intercept,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            max_iter=self.max_iter,
            tol=self.tol,
            linesearch_max_iter=self.linesearch_max_iter,
            verbose=self.verbose,
            handle=self.handle,
        )

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = np.asarray([n_iter])

        return self

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        },
    )
    @cuml.internals.api_base_return_any_skipall
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        scores = self.decision_function(
            X, convert_dtype=convert_dtype
        ).to_output("cupy")

        # Choose a smaller index type when possible
        nclasses = len(self.classes_)
        ind_dtype = (
            np.int32 if nclasses <= np.iinfo(np.int32).max else np.int64
        )

        if scores.ndim == 1:
            indices = (scores > 0).astype(ind_dtype)
        else:
            indices = cp.argmax(scores, axis=1)

        # TODO: Scikit-Learn's `LogisticRegression.predict` returns the same
        # dtype as the input classes, _and_ natively supports non-numeric
        # dtypes. CumlArray doesn't currently support wrapping containers with
        # non-numeric dtypes. As such, we cannot rely on the normal decorators
        # to handle output conversion, and need to handle output coercion
        # internally. This is a hack.
        output_type = self._get_output_type(X)

        if self.classes_.dtype.kind in "ifu":
            if (self.classes_ == np.arange(nclasses)).all():
                # Fast path for common case of monotonically increasing numeric classes
                out = indices.astype(self.classes_.dtype)
            else:
                # Classes are not monotonically increasing from 0, we need to
                # do a transform.
                out = cp.asarray(self.classes_).take(indices)

            # Numeric types can always rely on CumlArray's output_type handling
            return CumlArray(out).to_output(output_type)
        else:
            # Non-numeric classes. We use cudf since it supports all types, and will
            # error appropriately later on when converting to outputs like `cupy`
            # that don't support strings.
            out = (
                cudf.Series(self.classes_).take(indices).reset_index(drop=True)
            )
            if output_type in ("cudf", "df_obj", "series"):
                return out
            elif output_type == "dataframe":
                return out.to_frame()
            elif output_type == "pandas":
                return cudf_to_pandas(out)
            elif output_type in ("numpy", "array"):
                return out.to_numpy(dtype=self.classes_.dtype)
            else:
                raise TypeError(
                    f"{output_type=} doesn't support objects of dtype {self.classes_.dtype!r}"
                )

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "probs",
            "type": "dense",
            "description": "Probabilities per class for each sample.",
            "shape": "(n_samples, n_classes)",
        },
    )
    def predict_proba(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the class probabilities for each class in X
        """
        n_classes = self.classes_.shape[0]

        scores = self.decision_function(X, convert_dtype=convert_dtype)
        scores = scores.to_output("cupy")
        if n_classes == 2:
            proba = cp.zeros((scores.shape[0], 2))
            proba[:, 1] = 1 / (1 + cp.exp(-scores.ravel()))
            proba[:, 0] = 1 - proba[:, 1]
        elif n_classes > 2:
            max_scores = cp.max(scores, axis=1).reshape((-1, 1))
            scores -= max_scores
            proba = cp.exp(scores)
            row_sum = cp.sum(proba, axis=1).reshape((-1, 1))
            proba /= row_sum
        return CumlArray(data=proba)

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "probs",
            "type": "dense",
            "description": "Log probabilities per class for each sample.",
            "shape": "(n_samples, n_classes)",
        },
    )
    def predict_log_proba(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the log class probabilities for each class in X
        """
        out = self.predict_proba(X, convert_dtype=convert_dtype).to_output(
            "cupy"
        )
        cp.log(out, out=out)
        return CumlArray(data=out)
