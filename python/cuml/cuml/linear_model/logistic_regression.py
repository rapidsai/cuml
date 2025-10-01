#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import pprint

import cupy as cp
import numpy as np

import cuml.internals
from cuml.common import using_output_type
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.internals import logger
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import (
    ClassifierMixin,
    FMajorInputTagMixin,
    SparseInputTagMixin,
)
from cuml.internals.output_utils import cudf_to_pandas
from cuml.preprocessing import LabelEncoder
from cuml.solvers import QN

supported_penalties = ["l1", "l2", None, "elasticnet"]

supported_solvers = ["qn"]


class LogisticRegression(
    Base,
    InteropMixin,
    ClassifierMixin,
    FMajorInputTagMixin,
    SparseInputTagMixin,
):
    """
    LogisticRegression is a linear model that is used to model probability of
    occurrence of certain events, for example probability of success or fail of
    an event.

    cuML's LogisticRegression can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or `__cuda_array_interface__`
    compliant), in addition to cuDF objects.
    It provides both single-class (using sigmoid loss) and multiple-class
    (using softmax loss) variants, depending on the input variables

    Only one solver option is currently available: Quasi-Newton (QN)
    algorithms. Even though it is presented as a single option, this solver
    resolves to two different algorithms underneath:

      - Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
        regularization

      - Limited Memory BFGS (L-BFGS) otherwise.


    Note that, just like in Scikit-learn, the bias will not be regularized.

    Examples
    --------

    .. code-block:: python

        >>> import cudf
        >>> import numpy as np

        >>> # Both import methods supported
        >>> # from cuml import LogisticRegression
        >>> from cuml.linear_model import LogisticRegression

        >>> X = cudf.DataFrame()
        >>> X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        >>> X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        >>> y = cudf.Series(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

        >>> reg = LogisticRegression()
        >>> reg.fit(X,y)
        LogisticRegression()
        >>> print(reg.coef_)
                 0         1
        0  0.69861  0.570058
        >>> print(reg.intercept_)
        0   -2.188...
        dtype: float32

        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = np.array([1,5], dtype = np.float32)
        >>> X_new['col2'] = np.array([2,5], dtype = np.float32)

        >>> preds = reg.predict(X_new)

        >>> print(preds)
        0    0.0
        1    1.0
        dtype: float32

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None} (default = 'l2')
        Used to specify the norm used in the penalization.
        If None or 'l2' are selected, then L-BFGS solver will be used.
        If 'l1' is selected, solver OWL-QN will be used.
        If 'elasticnet' is selected, OWL-QN will be used if l1_ratio > 0,
        otherwise L-BFGS will be used.
    tol : float (default = 1e-4)
        Tolerance for stopping criteria.
        The exact stopping conditions depend on the chosen solver.
        Check the solver's documentation for more details:

          * :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`

    C : float (default = 1.0)
        Inverse of regularization strength; must be a positive float.
    fit_intercept : boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    class_weight : dict or 'balanced', default=None
        By default all classes have a weight one. However, a dictionary
        can be provided with weights associated with classes
        in the form ``{class_label: weight}``. The "balanced" mode
        uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``. Note that
        these weights will be multiplied with sample_weight
        (passed through the fit method) if sample_weight is specified.
    max_iter : int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    linesearch_max_iter : int (default = 50)
        Max number of linesearch iterations per outer iteration used in the
        lbfgs and owl QN solvers.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    l1_ratio : float or None, optional (default=None)
        The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`
    solver : 'qn' (default='qn')
        Algorithm to use in the optimization problem. Currently only `qn` is
        supported, which automatically selects either L-BFGS or OWL-QN
        depending on the conditions of the l1 regularization described
        above.
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
    coef_: dev array, dim (n_classes, n_features) or (n_classes, n_features+1)
        The estimated coefficients for the logistic regression model.
    intercept_: device array (n_classes, 1)
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
    """

    class_weight = CumlArrayDescriptor(order="F")
    expl_spec_weights_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.linear_model.LogisticRegression"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
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
        verbose=False,
        l1_ratio=None,
        solver="qn",
        handle=None,
        output_type=None,
    ):

        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        if penalty not in supported_penalties:
            raise ValueError("`penalty` " + str(penalty) + " not supported.")

        if solver not in supported_solvers:
            raise ValueError(
                "Only quasi-newton `qn` solver is "
                " supported, not %s" % solver
            )
        self.solver = solver
        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.l1_ratio = None
        if self.penalty == "elasticnet":
            if l1_ratio is None:
                raise ValueError(
                    "l1_ratio has to be specified for" "loss='elasticnet'"
                )
            if l1_ratio < 0.0 or l1_ratio > 1.0:
                msg = "l1_ratio value has to be between 0.0 and 1.0"
                raise ValueError(msg.format(l1_ratio))
            self.l1_ratio = l1_ratio

        l1_strength, l2_strength = self._get_qn_params()

        loss = "sigmoid"

        if class_weight is not None:
            self._build_class_weights(class_weight)
        else:
            self.class_weight = None

        self.solver_model = QN(
            loss=loss,
            fit_intercept=self.fit_intercept,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            max_iter=self.max_iter,
            linesearch_max_iter=self.linesearch_max_iter,
            tol=self.tol,
            verbose=self.verbose,
            handle=self.handle,
        )

        if logger.should_log_for(logger.level_enum.debug):
            self.verb_prefix = "CY::"
            logger.debug(self.verb_prefix + "Estimator parameters:")
            logger.debug(pprint.pformat(self.__dict__))
        else:
            self.verb_prefix = ""

    @generate_docstring(X="dense_sparse")
    @cuml.internals.api_base_return_any()
    def fit(
        self, X, y, sample_weight=None, *, convert_dtype=True
    ) -> "LogisticRegression":
        """
        Fit the model with X and y.

        """
        if is_sparse(X):
            # Skip conversion of sparse arrays since sparse are already in the
            # correct format and don't need array-like input handling. The
            # conversion will be handled by the solver model.
            n_rows, self.n_features_in_, self.dtype = (
                X.shape[0],
                (X.shape[1] if X.ndim == 2 else 1),
                X.dtype,
            )
        else:
            X, n_rows, self.n_features_in_, self.dtype = input_to_cuml_array(
                X,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                check_dtype=[np.float32, np.float64],
                order="K",
            )

        if hasattr(X, "index"):
            self.feature_names_in_ = X.index

        # LabelEncoder currently always returns cudf (ignoring output_type),
        # but we need to explicitly set `output_type` or we'll get an error
        # on init since `output_type` would default to `mirror`.
        enc = LabelEncoder(output_type="cudf")
        y_orig_dtype = getattr(y, "dtype", None)
        y = enc.fit_transform(y).to_cupy()
        if y_orig_dtype is None:
            y_orig_dtype = y.dtype
        n_rows = len(y)
        classes = enc.classes_.to_numpy()

        # TODO: LabelEncoder doesn't currently map dtypes the same way as it
        # does in scikit-learn. Until that's fixed we fix them up here.
        if y_orig_dtype.kind == "U":
            classes = classes.astype("U")
        elif y_orig_dtype == "float16":
            classes = classes.astype("float16")
        self.classes_ = classes

        self._num_classes = len(self.classes_)

        if sample_weight is not None or self.class_weight is not None:
            if sample_weight is None:
                sample_weight = cp.ones(n_rows)

            sample_weight, n_weights, D, _ = input_to_cuml_array(sample_weight)

            if n_rows != n_weights or D != 1:
                raise ValueError(
                    "sample_weight.shape == {}, "
                    "expected ({},)!".format(sample_weight.shape, n_rows)
                )

            def check_expl_spec_weights():
                with cuml.using_output_type("numpy"):
                    for c in self.expl_spec_weights_:
                        i = np.searchsorted(self.classes_, c)
                        if i >= self._num_classes or self.classes_[i] != c:
                            msg = "Class label {} not present.".format(c)
                            raise ValueError(msg)

            if self.class_weight is not None:
                if self.class_weight == "balanced":
                    class_weight = n_rows / (
                        self._num_classes * cp.bincount(y)
                    )
                    class_weight = CumlArray(class_weight)
                else:
                    check_expl_spec_weights()
                    n_explicit = self.class_weight.shape[0]
                    if n_explicit != self._num_classes:
                        class_weight = cp.ones(self._num_classes)
                        class_weight[:n_explicit] = self.class_weight
                        class_weight = CumlArray(class_weight)
                        self.class_weight = class_weight
                    else:
                        class_weight = self.class_weight
                sample_weight *= class_weight[y].to_output("cupy")
                sample_weight = CumlArray(sample_weight)

        if self._num_classes > 2:
            loss = "softmax"
        else:
            loss = "sigmoid"

        if logger.should_log_for(logger.level_enum.debug):
            logger.debug(self.verb_prefix + "Setting loss to " + str(loss))

        self.solver_model.loss = loss

        if logger.should_log_for(logger.level_enum.debug):
            logger.debug(self.verb_prefix + "Calling QN fit " + str(loss))

        self.solver_model.fit(
            X, y, sample_weight=sample_weight, convert_dtype=convert_dtype
        )

        self.n_iter_ = np.asarray([self.solver_model.num_iters])

        # coefficients and intercept are contained in the same array
        if logger.should_log_for(logger.level_enum.debug):
            logger.debug(
                self.verb_prefix + "Setting coefficients " + str(loss)
            )

        if logger.should_log_for(logger.level_enum.trace):
            with using_output_type("cupy"):
                logger.trace(
                    self.verb_prefix
                    + "Coefficients: "
                    + str(self.solver_model.coef_)
                )
                if self.fit_intercept:
                    logger.trace(
                        self.verb_prefix
                        + "Intercept: "
                        + str(self.solver_model.intercept_)
                    )

        return self

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "score",
            "type": "dense",
            "description": "Confidence score",
            "shape": "(n_samples, n_classes)",
        },
    )
    def decision_function(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Gives confidence score for X

        """
        return self.solver_model._decision_function(
            X, convert_dtype=convert_dtype
        )

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        },
    )
    @cuml.internals.api_base_return_any()
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        indices = self.solver_model.predict(X, convert_dtype=convert_dtype)

        # TODO: Scikit-Learn's `LogisticRegression.predict` returns the same
        # dtype as the input classes, _and_ natively supports non-numeric
        # dtypes. CumlArray doesn't currently support wrapping containers with
        # non-numeric dtypes. As such, we cannot rely on the normal decorators
        # to handle output conversion, and need to handle output coercion
        # internally. This is a hack.
        output_type = self._get_output_type(X)

        is_numeric = self.classes_.dtype.kind in "ifu"
        nclasses = len(self.classes_)

        # Choose a smaller index type when possible
        ind_dtype = (
            np.int32 if nclasses <= np.iinfo(np.int32).max else np.int64
        )

        if is_numeric:
            if (self.classes_ == np.arange(nclasses)).all():
                # Fast path for common case of monotonically increasing numeric classes
                out = indices.to_output(
                    "cupy", output_dtype=self.classes_.dtype
                )
            else:
                # Classes are not monotonically increasing from 0, we need to
                # do a transform.
                out = cp.asarray(self.classes_).take(
                    indices.to_output("cupy", output_dtype=ind_dtype)
                )

            # Numeric types can always rely on CumlArray's output_type handling
            return CumlArray(out).to_output(output_type)
        else:
            # Non-numeric classes. We use cudf since it supports all types, and will
            # error appropriately later on when converting to outputs like `cupy`
            # that don't support strings.
            import cudf

            out = (
                cudf.Series(self.classes_)
                .take(indices.to_output("cupy", output_dtype=ind_dtype))
                .reset_index(drop=True)
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
            "name": "preds",
            "type": "dense",
            "description": "Predicted class \
                                                       probabilities",
            "shape": "(n_samples, n_classes)",
        },
    )
    def predict_proba(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the class probabilities for each class in X
        """
        return self._predict_proba_impl(
            X, convert_dtype=convert_dtype, log_proba=False
        )

    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Logaright of predicted \
                                                       class probabilities",
            "shape": "(n_samples, n_classes)",
        },
    )
    def predict_log_proba(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts the log class probabilities for each class in X

        """
        return self._predict_proba_impl(
            X, convert_dtype=convert_dtype, log_proba=True
        )

    def _predict_proba_impl(
        self, X, convert_dtype=False, log_proba=False
    ) -> CumlArray:
        _num_classes = self.classes_.shape[0]

        scores = self.decision_function(
            X, convert_dtype=convert_dtype
        ).to_output("cupy")
        if _num_classes == 2:
            proba = cp.zeros((scores.shape[0], 2))
            proba[:, 1] = 1 / (1 + cp.exp(-scores.ravel()))
            proba[:, 0] = 1 - proba[:, 1]
        elif _num_classes > 2:
            max_scores = cp.max(scores, axis=1).reshape((-1, 1))
            scores -= max_scores
            proba = cp.exp(scores)
            row_sum = cp.sum(proba, axis=1).reshape((-1, 1))
            proba /= row_sum

        if log_proba:
            proba = cp.log(proba)

        return proba

    def _get_qn_params(self):
        if self.penalty is None:
            l1_strength = 0.0
            l2_strength = 0.0

        elif self.penalty == "l1":
            l1_strength = 1.0 / self.C
            l2_strength = 0.0

        elif self.penalty == "l2":
            l1_strength = 0.0
            l2_strength = 1.0 / self.C

        else:
            strength = 1.0 / self.C
            l1_strength = self.l1_ratio * strength
            l2_strength = (1.0 - self.l1_ratio) * strength
        return l1_strength, l2_strength

    def _build_class_weights(self, class_weight):
        if class_weight is None:
            self.class_weight = None
        elif class_weight == "balanced":
            self.class_weight = "balanced"
        else:
            classes = list(class_weight.keys())
            weights = list(class_weight.values())
            max_class = sorted(classes)[-1]
            class_weight = cp.ones(max_class + 1)
            class_weight[classes] = weights
            self.class_weight, _, _, _ = input_to_cuml_array(class_weight)
            self.expl_spec_weights_, _, _, _ = input_to_cuml_array(
                np.array(classes)
            )

    def set_params(self, **params):
        super().set_params(**params)
        rebuild_params = False
        # Remove class-specific parameters
        for param_name in ["penalty", "l1_ratio", "C"]:
            if param_name in params:
                params.pop(param_name)
                rebuild_params = True
        if rebuild_params:
            # re-build QN solver parameters
            l1_strength, l2_strength = self._get_qn_params()
            params.update(
                {"l1_strength": l1_strength, "l2_strength": l2_strength}
            )
        if "class_weight" in params:
            # re-build class weight
            class_weight = params.pop("class_weight")
            self._build_class_weights(class_weight)

        # if the user is setting the solver, then
        # it cannot be propagated to the solver model itself.
        _ = params.pop("solver", None)
        self.solver_model.set_params(**params)
        return self

    @property
    def coef_(self) -> CumlArray:
        return self.solver_model.coef_

    @coef_.setter
    def coef_(self, value):
        self.solver_model.coef_ = value

    @property
    def intercept_(self) -> CumlArray:
        return self.solver_model.intercept_

    @intercept_.setter
    def intercept_(self, value):
        self.solver_model.intercept_ = value
