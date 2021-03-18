#
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

import numpy as np
import cupy as cp
import pprint

import cuml.internals
from cuml.solvers import QN
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.array import CumlArray
from cuml.common.doc_utils import generate_docstring
import cuml.common.logger as logger
from cuml.common import input_to_cuml_array
from cuml.common import using_output_type
from cuml.common.mixins import FMajorInputTagMixin


supported_penalties = ["l1", "l2", "none", "elasticnet"]

supported_solvers = ["qn"]


class LogisticRegression(Base,
                         ClassifierMixin,
                         FMajorInputTagMixin):
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

        import cudf
        import numpy as np

        # Both import methods supported
        # from cuml import LogisticRegression
        from cuml.linear_model import LogisticRegression

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series( np.array([0.0, 0.0, 1.0, 1.0], dtype = np.float32) )

        reg = LogisticRegression()
        reg.fit(X,y)

        print("Coefficients:")
        print(reg.coef_)
        print("Intercept:")
        print(reg.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([1,5], dtype = np.float32)
        X_new['col2'] = np.array([2,5], dtype = np.float32)

        preds = reg.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python

        Coefficients:
                    0.22309814
                    0.21012752
        Intercept:
                    -0.7548761
        Predictions:
                    0    0.0
                    1    1.0

    Parameters
    -----------
    penalty: 'none', 'l1', 'l2', 'elasticnet' (default = 'l2')
        Used to specify the norm used in the penalization.
        If 'none' or 'l2' are selected, then L-BFGS solver will be used.
        If 'l1' is selected, solver OWL-QN will be used.
        If 'elasticnet' is selected, OWL-QN will be used if l1_ratio > 0,
        otherwise L-BFGS will be used.
    tol: float (default = 1e-4)
       The training process will stop if current_loss > previous_loss - tol
    C: float (default = 1.0)
       Inverse of regularization strength; must be a positive float.
    fit_intercept: boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    class_weight: None
        Custom class weighs are currently not supported.
    class_weight: dict or 'balanced', default=None
        By default all classes have a weight one. However, a dictionary
        can be provided with weights associated with classes
        in the form ``{class_label: weight}``. The "balanced" mode
        uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``. Note that
        these weights will be multiplied with sample_weight
        (passed through the fit method) if sample_weight is specified.
    max_iter: int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    linesearch_max_iter: int (default = 50)
        Max number of linesearch iterations per outer iteration used in the
        lbfgs and owl QN solvers.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    l1_ratio: float or None, optional (default=None)
        The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`
    solver: 'qn', 'lbfgs', 'owl' (default='qn').
        Algorithm to use in the optimization problem. Currently only `qn` is
        supported, which automatically selects either L-BFGS or OWL-QN
        depending on the conditions of the l1 regularization described
        above. Options 'lbfgs' and 'owl' are just convenience values that
        end up using the same solver following the same rules.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    -----------
    coef_: dev array, dim (n_classes, n_features) or (n_classes, n_features+1)
        The estimated coefficients for the linear regression model.
        Note: this includes the intercept as the last column if fit_intercept
        is True
    intercept_: device array (n_classes, 1)
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    ------

    cuML's LogisticRegression uses a different solver that the equivalent
    Scikit-learn, except when there is no penalty and `solver=lbfgs` is
    used in Scikit-learn. This can cause (smaller) differences in the
    coefficients and predictions of the model, similar to
    using different solvers in Scikit-learn.

    For additional information, see Scikit-learn's LogistRegression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """

    classes_ = CumlArrayDescriptor()
    class_weight_ = CumlArrayDescriptor()
    expl_spec_weights_ = CumlArrayDescriptor()

    def __init__(
        self,
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

        super(LogisticRegression, self).__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        if penalty not in supported_penalties:
            raise ValueError("`penalty` " + str(penalty) + "not supported.")

        if solver not in supported_solvers:
            raise ValueError("Only quasi-newton `qn` solver is "
                             " supported, not %s" % solver)
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

        if self.penalty == "none":
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

        loss = "sigmoid"

        if class_weight is not None:
            if class_weight == 'balanced':
                self.class_weight_ = 'balanced'
            else:
                classes = list(class_weight.keys())
                weights = list(class_weight.values())
                max_class = sorted(classes)[-1]
                class_weight = cp.ones(max_class + 1)
                class_weight[classes] = weights
                self.class_weight_, _, _, _ = input_to_cuml_array(class_weight)
                self.expl_spec_weights_, _, _, _ = \
                    input_to_cuml_array(np.array(classes))
        else:
            self.class_weight_ = None

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

        if logger.should_log_for(logger.level_debug):
            self.verb_prefix = "CY::"
            logger.debug(self.verb_prefix + "Estimator parameters:")
            logger.debug(pprint.pformat(self.__dict__))
        else:
            self.verb_prefix = ""

    @generate_docstring()
    @cuml.internals.api_base_return_any(set_output_dtype=True)
    def fit(self, X, y, sample_weight=None,
            convert_dtype=True) -> "LogisticRegression":
        """
        Fit the model with X and y.

        """
        # Converting y to device array here to use `unique` function
        # since calling input_to_cuml_array again in QN has no cost
        # Not needed to check dtype since qn class checks it already
        y_m, n_rows, _, _ = input_to_cuml_array(y)
        self.classes_ = cp.unique(y_m)
        self._num_classes = len(self.classes_)

        if self._num_classes == 2:
            if self.classes_[0] != 0 or self.classes_[1] != 1:
                raise ValueError("Only values of 0 and 1 are"
                                 " supported for binary classification.")

        if sample_weight is not None or self.class_weight_ is not None:
            if sample_weight is None:
                sample_weight = cp.ones(n_rows)

            sample_weight, n_weights, D, _ = input_to_cuml_array(sample_weight)

            if n_rows != n_weights or D != 1:
                raise ValueError("sample_weight.shape == {}, "
                                 "expected ({},)!".format(sample_weight.shape,
                                                          n_rows))

            def check_expl_spec_weights():
                with cuml.using_output_type("numpy"):
                    for c in self.expl_spec_weights_:
                        i = np.searchsorted(self.classes_, c)
                        if i >= self._num_classes or self.classes_[i] != c:
                            msg = "Class label {} not present.".format(c)
                            raise ValueError(msg)

            if self.class_weight_ is not None:
                if self.class_weight_ == 'balanced':
                    class_weight = n_rows / \
                                   (self._num_classes *
                                    cp.bincount(y_m.to_output('cupy')))
                    class_weight = CumlArray(class_weight)
                else:
                    check_expl_spec_weights()
                    n_explicit = self.class_weight_.shape[0]
                    if n_explicit != self._num_classes:
                        class_weight = cp.ones(self._num_classes)
                        class_weight[:n_explicit] = self.class_weight_
                        class_weight = CumlArray(class_weight)
                        self.class_weight_ = class_weight
                    else:
                        class_weight = self.class_weight_
                out = y_m.to_output('cupy')
                sample_weight *= class_weight[out].to_output('cupy')
                sample_weight = CumlArray(sample_weight)

        if self._num_classes > 2:
            loss = "softmax"
        else:
            loss = "sigmoid"

        if logger.should_log_for(logger.level_debug):
            logger.debug(self.verb_prefix + "Setting loss to " + str(loss))

        self.solver_model.loss = loss

        if logger.should_log_for(logger.level_debug):
            logger.debug(self.verb_prefix + "Calling QN fit " + str(loss))

        self.solver_model.fit(X, y_m, sample_weight=sample_weight,
                              convert_dtype=convert_dtype)

        # coefficients and intercept are contained in the same array
        if logger.should_log_for(logger.level_debug):
            logger.debug(
                self.verb_prefix + "Setting coefficients " + str(loss)
            )

        if logger.should_log_for(logger.level_trace):
            with using_output_type("cupy"):
                logger.trace(self.verb_prefix + "Coefficients: " +
                             str(self.solver_model.coef_))
                if self.fit_intercept:
                    logger.trace(
                        self.verb_prefix
                        + "Intercept: "
                        + str(self.solver_model.intercept_)
                    )

        return self

    @generate_docstring(return_values={'name': 'score',
                                       'type': 'dense',
                                       'description': 'Confidence score',
                                       'shape': '(n_samples, n_classes)'})
    def decision_function(self, X, convert_dtype=False) -> CumlArray:
        """
        Gives confidence score for X

        """
        return self.solver_model._decision_function(
            X,
            convert_dtype=convert_dtype
        )

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        return self.solver_model.predict(X, convert_dtype=convert_dtype)

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted class \
                                                       probabilities',
                                       'shape': '(n_samples, n_classes)'})
    def predict_proba(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the class probabilities for each class in X

        """
        return self._predict_proba_impl(
            X,
            convert_dtype=convert_dtype,
            log_proba=False
        )

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Logaright of predicted \
                                                       class probabilities',
                                       'shape': '(n_samples, n_classes)'})
    def predict_log_proba(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the log class probabilities for each class in X

        """
        return self._predict_proba_impl(
            X,
            convert_dtype=convert_dtype,
            log_proba=True
        )

    def _predict_proba_impl(self,
                            X,
                            convert_dtype=False,
                            log_proba=False) -> CumlArray:
        # TODO:
        # We currently need to grab the dtype and ncols attributes via the
        # qn solver due to https://github.com/rapidsai/cuml/issues/2404
        X_m, _, _, self.dtype = input_to_cuml_array(
            X,
            check_dtype=self.solver_model.dtype,
            convert_to_dtype=(
                self.solver_model.dtype if convert_dtype else None
            ),
            check_cols=self.solver_model.n_cols,
        )

        scores = cp.asarray(
            self.decision_function(X_m, convert_dtype=convert_dtype), order="F"
        ).T
        if self._num_classes == 2:
            proba = cp.zeros((scores.shape[0], 2))
            proba[:, 1] = 1 / (1 + cp.exp(-scores.ravel()))
            proba[:, 0] = 1 - proba[:, 1]
        elif self._num_classes > 2:
            max_scores = cp.max(scores, axis=1).reshape((-1, 1))
            scores -= max_scores
            proba = cp.exp(scores)
            row_sum = cp.sum(proba, axis=1).reshape((-1, 1))
            proba /= row_sum

        if log_proba:
            proba = cp.log(proba)

        return proba

    def get_param_names(self):
        return super().get_param_names() + [
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

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        super(LogisticRegression, self).__init__(handle=None,
                                                 verbose=state["verbose"])
        self.__dict__.update(state)
