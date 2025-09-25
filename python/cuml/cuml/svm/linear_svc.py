# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import numpy as np

from cuml.common import input_to_cuml_array
from cuml.internals.array import CumlArray
from cuml.internals.interop import UnsupportedOnGPU, to_cpu, to_gpu
from cuml.internals.mixins import ClassifierMixin
from cuml.svm.linear import LinearSVM, LinearSVM_defaults  # noqa: F401
from cuml.svm.svc import apply_class_weight

__all__ = ["LinearSVC"]


class LinearSVC(LinearSVM, ClassifierMixin):
    """
    LinearSVC (Support Vector Classification with the linear kernel)

    Construct a linear SVM classifier for training and predictions.

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.svm import LinearSVC
        >>> X = cp.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
        ...              dtype=cp.float32);
        >>> y = cp.array([0, 0, 1, 0, 1, 1], dtype=cp.float32)
        >>> clf = LinearSVC(loss='squared_hinge', penalty='l1', C=1)
        >>> clf.fit(X, y)
        LinearSVC()
        >>> print("Predicted labels:", clf.predict(X))
        Predicted labels: [0 0 1 0 1 1]

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    penalty : {{'l1', 'l2'}} (default = '{LinearSVM_defaults.penalty}')
        The regularization term of the target function.
    loss : {LinearSVC.REGISTERED_LOSSES} (default = 'squared_hinge')
        The loss term of the target function.
    fit_intercept : {LinearSVM_defaults.fit_intercept.__class__.__name__ \
            } (default = {LinearSVM_defaults.fit_intercept})
        Whether to fit the bias term. Set to False if you expect that the
        data is already centered.
    penalized_intercept : { \
            LinearSVM_defaults.penalized_intercept.__class__.__name__ \
            } (default = {LinearSVM_defaults.penalized_intercept})
        When true, the bias term is treated the same way as other features;
        i.e. it's penalized by the regularization term of the target function.
        Enabling this feature forces an extra copying the input data X.
    max_iter : {LinearSVM_defaults.max_iter.__class__.__name__ \
            } (default = {LinearSVM_defaults.max_iter})
        Maximum number of iterations for the underlying solver.
    linesearch_max_iter : { \
            LinearSVM_defaults.linesearch_max_iter.__class__.__name__ \
            } (default = {LinearSVM_defaults.linesearch_max_iter})
        Maximum number of linesearch (inner loop) iterations for
        the underlying (QN) solver.
    lbfgs_memory : { \
            LinearSVM_defaults.lbfgs_memory.__class__.__name__ \
            } (default = {LinearSVM_defaults.lbfgs_memory})
        Number of vectors approximating the hessian for the underlying QN
        solver (l-bfgs).
    class_weight : dict or string (default=None)
        Weights to modify the parameter C for class i to class_weight[i]*C. The
        string 'balanced' is also accepted, in which case ``class_weight[i] =
        n_samples / (n_classes * n_samples_of_class[i])``
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    C : {LinearSVM_defaults.C.__class__.__name__ \
            } (default = {LinearSVM_defaults.C})
        The constant scaling factor of the loss term in the target formula
          `F(X, y) = penalty(X) + C * loss(X, y)`.
    grad_tol : {LinearSVM_defaults.grad_tol.__class__.__name__ \
            } (default = {LinearSVM_defaults.grad_tol})
        The threshold on the gradient for the underlying QN solver.
    change_tol : {LinearSVM_defaults.change_tol.__class__.__name__ \
            } (default = {LinearSVM_defaults.change_tol})
        The threshold on the function change for the underlying QN solver.
    tol : Optional[float] (default = None)
        Tolerance for the stopping criterion.
        This is a helper transient parameter that, when present, sets both
        `grad_tol` and `change_tol` to the same value. When any of the two
        `***_tol` parameters are passed as well, they take the precedence.
    probability: {LinearSVM_defaults.probability.__class__.__name__ \
            } (default = {LinearSVM_defaults.probability})
        Enable or disable probability estimates.
    multi_class : {{currently, only 'ovr'}} (default = 'ovr')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_
    output_type : {{'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    intercept_ : float, shape (`n_classes_`,)
        The constant in the decision function
    coef_ : float, shape (`n_classes_`, n_cols)
        The vectors defining the hyperplanes that separate the classes.
    classes_ : float, shape (`n_classes_`,)
        Array of class labels.
    probScale_ : float, shape (`n_classes_`, 2)
        Probability calibration constants (for the probabolistic output).
    n_classes_ : int
        Number of classes

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVC>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's LinearSVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.
    """

    _cpu_class_path = "sklearn.svm.LinearSVC"
    REGISTERED_LOSSES = set(["hinge", "squared_hinge"])

    def __init__(self, **kwargs):
        # NB: the keyword arguments are filtered in python/cuml/svm/linear.pyx
        #     the default parameter values are reexported from
        #                                      cpp/include/cuml/svm/linear.hpp
        # set classification-specific defaults
        if "loss" not in kwargs:
            kwargs["loss"] = "squared_hinge"
        if "multi_class" not in kwargs:
            # 'multi_class' is a real parameter here
            # 'multiclass_strategy' is an ephemeral compatibility parameter
            #              for easier switching between
            #              sklearn.LinearSVC <-> cuml.LinearSVC <-> cuml.SVC
            kwargs["multi_class"] = kwargs.pop("multiclass_strategy", "ovr")

        super().__init__(**kwargs)

    @classmethod
    def _params_from_cpu(cls, model):
        if model.multi_class != "ovr":
            raise UnsupportedOnGPU(
                f"`multi_class={model.multi_class}` is not supported"
            )

        params = {
            "loss": model.loss,
            "multi_class": model.multi_class,
            "class_weight": model.class_weight,
            "penalty": model.penalty,
            **super()._params_from_cpu(model),
        }

        return params

    def _params_to_cpu(self):
        return {
            "loss": self.loss,
            "multi_class": self.multi_class,
            "class_weight": self.class_weight,
            "penalty": self.penalty,
            **super()._params_to_cpu(),
        }

    def _attrs_from_cpu(self, model):
        return {
            "classes_": to_gpu(model.classes_, order="F", dtype=np.float64),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "classes_": to_cpu(self.classes_, order="C"),
            **super()._attrs_to_cpu(model),
        }

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Classification loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'."
            )
        self.__loss = loss

    @classmethod
    def _get_param_names(cls):
        return list(
            {
                "handle",
                "class_weight",
                "verbose",
                "penalty",
                "loss",
                "fit_intercept",
                "penalized_intercept",
                "probability",
                "max_iter",
                "linesearch_max_iter",
                "lbfgs_memory",
                "C",
                "grad_tol",
                "change_tol",
                "multi_class",
            }.union(super()._get_param_names())
        )

    def fit(
        self, X, y, sample_weight=None, *, convert_dtype=True
    ) -> "LinearSVM":
        X, n_rows, self.n_features_in_, self.dtype = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        )

        convert_to_dtype = self.dtype if convert_dtype else None
        y = input_to_cuml_array(
            y,
            check_dtype=self.dtype,
            convert_to_dtype=convert_to_dtype,
            check_rows=n_rows,
            check_cols=1,
        ).array

        if X.size == 0 or y.size == 0:
            raise ValueError("empty data")

        sample_weight = apply_class_weight(
            self.handle,
            sample_weight,
            self.class_weight,
            y,
            self.verbose,
            self.output_type,
            X.dtype,
        )
        return super(LinearSVC, self).fit(
            X, y, sample_weight, convert_dtype=convert_dtype
        )

    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        y_pred = super().predict(X, convert_dtype=convert_dtype)
        # Cast to int64 to match expected classifier interface
        return y_pred.to_output("cupy", output_dtype="int64")
