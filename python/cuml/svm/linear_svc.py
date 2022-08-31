# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from cuml.common.mixins import ClassifierMixin
from cuml.svm.linear import LinearSVM, LinearSVM_defaults  # noqa: F401

__all__ = ['LinearSVC']


class LinearSVC(LinearSVM, ClassifierMixin):
    '''
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
        Predicted labels: [0. 0. 1. 0. 1. 1.]

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
    output_type : {{'input', 'cudf', 'cupy', 'numpy', 'numba'}}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

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
    '''

    REGISTERED_LOSSES = set([
        'hinge',
        'squared_hinge'])

    def __init__(self, **kwargs):
        # NB: the keyword arguments are filtered in python/cuml/svm/linear.pyx
        #     the default parameter values are reexported from
        #                                      cpp/include/cuml/svm/linear.hpp
        # set classification-specific defaults
        if 'loss' not in kwargs:
            kwargs['loss'] = 'squared_hinge'
        if 'multi_class' not in kwargs:
            # 'multi_class' is a real parameter here
            # 'multiclass_strategy' is an ephemeral compatibility parameter
            #              for easier switching between
            #              sklearn.LinearSVC <-> cuml.LinearSVC <-> cuml.SVC
            kwargs['multi_class'] = kwargs.pop('multiclass_strategy', 'ovr')

        super().__init__(**kwargs)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Classification loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        self.__loss = loss

    def get_param_names(self):
        return list({
            "handle",
            "verbose",
            'penalty',
            'loss',
            'fit_intercept',
            'penalized_intercept',
            'probability',
            'max_iter',
            'linesearch_max_iter',
            'lbfgs_memory',
            'C',
            'grad_tol',
            'change_tol',
            'multi_class',
        }.union(super().get_param_names()))
