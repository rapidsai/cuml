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

from cuml.internals.mixins import RegressorMixin
from cuml.svm.linear import LinearSVM, LinearSVM_defaults  # noqa: F401

__all__ = ["LinearSVR"]


class LinearSVR(LinearSVM, RegressorMixin):
    """
    LinearSVR (Support Vector Regression with the linear kernel)

    Construct a linear SVM regressor for training and predictions.

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.svm import LinearSVR
        >>> X = cp.array([[1], [2], [3], [4], [5]], dtype=cp.float32)
        >>> y = cp.array([1.1, 4, 5, 3.9, 8.], dtype=cp.float32)
        >>> reg = LinearSVR(loss='epsilon_insensitive', C=10,
        ...                 epsilon=0.1, verbose=0)
        >>> reg.fit(X, y)
        LinearSVR()
        >>> print("Predicted values:", reg.predict(X)) # doctest: +SKIP
        Predicted labels: [1.8993504 3.3995128 4.899675  6.399837  7.899999]

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
    loss : {LinearSVR.REGISTERED_LOSSES} (default = 'epsilon_insensitive')
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
    epsilon : {LinearSVM_defaults.epsilon.__class__.__name__ \
            } (default = {LinearSVM_defaults.epsilon})
        The epsilon-sensitivity parameter for the SVR loss function.
    output_type : {{'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    intercept_ : float, shape (1,)
        The constant in the decision function
    coef_ : float, shape (1, n_cols)
        The coefficients of the linear decision function.

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVR>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's LinearSVR
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html>`_.
    """

    _cpu_class_path = "sklearn.svm.LinearSVR"
    REGISTERED_LOSSES = set(
        ["epsilon_insensitive", "squared_epsilon_insensitive"]
    )

    def __init__(self, **kwargs):
        # NB: the keyword arguments are filtered in python/cuml/svm/linear.pyx
        #     the default parameter values are reexported from
        #                                      cpp/include/cuml/svm/linear.hpp
        # set regression-specific defaults
        if "loss" not in kwargs:
            kwargs["loss"] = "epsilon_insensitive"

        super().__init__(**kwargs)

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "loss": model.loss,
            "epsilon": model.epsilon,
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            "loss": self.loss,
            "epsilon": self.epsilon,
            **super()._params_to_cpu(),
        }

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Regression loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'."
            )
        self.__loss = loss

    @classmethod
    def _get_param_names(cls):
        return list(
            {
                "handle",
                "verbose",
                "penalty",
                "loss",
                "fit_intercept",
                "penalized_intercept",
                "max_iter",
                "linesearch_max_iter",
                "lbfgs_memory",
                "C",
                "grad_tol",
                "change_tol",
                "epsilon",
            }.union(super()._get_param_names())
        )
