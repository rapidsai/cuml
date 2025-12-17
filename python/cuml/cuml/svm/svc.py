# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.classification import (
    decode_labels,
    preprocess_labels,
    process_class_weight,
)
from cuml.common.doc_utils import generate_docstring
from cuml.common.exceptions import NotFittedError
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.input_utils import (
    input_to_cuml_array,
    input_to_host_array,
    input_to_host_array_with_sparse_support,
)
from cuml.internals.interop import UnsupportedOnCPU, UnsupportedOnGPU
from cuml.internals.logger import warn
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import (
    exit_internal_context,
    reflect,
    run_in_internal_context,
)
from cuml.internals.utils import check_random_seed
from cuml.multiclass import OneVsOneClassifier, OneVsRestClassifier
from cuml.svm.svm_base import SVMBase


class SVC(SVMBase, ClassifierMixin):
    """
    SVC (C-Support Vector Classification)

    Construct an SVC classifier for training and predictions.

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.svm import SVC
        >>> X = cp.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
        ...              dtype=cp.float32);
        >>> y = cp.array([-1, -1, 1, -1, 1, 1], dtype=cp.float32)
        >>> clf = SVC(kernel='poly', degree=2, gamma='auto', C=1)
        >>> clf.fit(X, y)
        SVC()
        >>> print("Predicted labels:", clf.predict(X))
        Predicted labels: [-1. -1.  1. -1.  1.  1.]

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    C : float (default = 1.0)
        Penalty parameter C
    kernel : string (default='rbf')
        Specifies the kernel function. Possible options: 'linear', 'poly',
        'rbf', 'sigmoid'. Currently precomputed kernels are not supported.
    degree : int (default=3)
        Degree of polynomial kernel function.
    gamma : float or string (default = 'scale')
        Coefficient for rbf, poly, and sigmoid kernels. You can specify the
        numeric value, or use one of the following options:

        - 'auto': gamma will be set to ``1 / n_features``
        - 'scale': gamma will be se to ``1 / (n_features * X.var())``

    coef0 : float (default = 0.0)
        Independent term in kernel function, only significant for poly and
        sigmoid
    tol : float (default = 1e-3)
        Tolerance for stopping criterion.
    cache_size : float (default = 1024.0)
        Size of the kernel cache during training in MiB. Increase it to improve
        the training time, at the cost of higher memory footprint. After
        training the kernel cache is deallocated.
        During prediction, we also need a temporary space to store kernel
        matrix elements (this can be significant if n_support is large).
        The cache_size variable sets an upper limit to the prediction
        buffer as well.
    class_weight : dict or string (default=None)
        Weights to modify the parameter C for class i to class_weight[i]*C. The
        string 'balanced' is also accepted, in which case ``class_weight[i] =
        n_samples / (n_classes * n_samples_of_class[i])``
    max_iter : int (default = -1)
        Limit the number of total iterations in the solver. Default of -1 for
        "no limit".
    decision_function_shape : str ('ovo' or 'ovr', default 'ovo')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_
    nochange_steps : int (default = 1000)
        We monitor how much our stopping criteria changes during outer
        iterations. If it does not change (changes less then 1e-3*tol)
        for nochange_steps consecutive steps, then we stop training.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    probability : bool (default = False)
        Set to ``True`` to enable probability estimates
        (``predict_proba``/``predict_log_proba``). Note that
        ``probability=True`` requires your training data have at least 5
        samples per class.
    random_state: int (default = None)
        Seed for random number generator (used only when ``probability=True``).
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    n_support_ : int
        The total number of support vectors. Note: this will change in the
        future to represent number support vectors for each class (like
        in Sklearn, see https://github.com/rapidsai/cuml/issues/956 )
    support_ : int, shape = (n_support)
        Device array of support vector indices
    support_vectors_ : float, shape (n_support, n_cols)
        Device array of support vectors
    dual_coef_ : float, shape = (1, n_support)
        Device array of coefficients for support vectors
    intercept_ : float
        The constant in the decision function
    fit_status_ : int
        0 if SVM is correctly fitted
    n_iter_ : array
        Number of outer iterations run by the solver for each model fit.
    coef_ : float, shape (1, n_cols)
        Only available for linear kernels. It is the normal of the
        hyperplane.
    classes_ : np.ndarray, shape=(n_classes,)
        A sorted array of the class labels.
    class_weight_ : np.ndarray of shape (n_classes,)
        Class weight multipliers, computed based on the ``class_weight``
        parameter.
    n_classes_ : int
        Number of classes

    Notes
    -----
    The solver uses the SMO method to fit the classifier. We use the Optimized
    Hierarchical Decomposition [1]_ variant of the SMO algorithm, similar to
    [2]_.

    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    References
    ----------
    .. [1] J. Vanek et al. A GPU-Architecture Optimized Hierarchical
       Decomposition Algorithm for Support VectorMachine Training, IEEE
       Transactions on Parallel and Distributed Systems, vol 28, no 12, 3330,
       (2017)

    .. [2] `Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs,
       Journal of Machine Learning Research, 19, 1-5 (2018)
       <https://github.com/Xtra-Computing/thundersvm>`_

    """

    _cpu_class_path = "sklearn.svm.SVC"

    @classmethod
    def _get_param_names(cls):
        params = super()._get_param_names()
        params.remove(
            "epsilon"
        )  # SVC doesn't expose `epsilon` in the constructor
        params.extend(
            [
                "probability",
                "random_state",
                "class_weight",
                "decision_function_shape",
            ]
        )
        return params

    @classmethod
    def _params_from_cpu(cls, model):
        params = super()._params_from_cpu(model)
        params.pop(
            "epsilon"
        )  # SVC doesn't expose `epsilon` in the constructor
        params.update(
            {
                "probability": model.probability,
                "random_state": model.random_state,
                "class_weight": model.class_weight,
                "decision_function_shape": model.decision_function_shape,
            }
        )
        return params

    def _params_to_cpu(self):
        params = super()._params_to_cpu()
        params.pop(
            "epsilon"
        )  # SVC doesn't expose `epsilon` in the constructor
        params.update(
            {
                "probability": self.probability,
                "random_state": self.random_state,
                "class_weight": self.class_weight,
                "decision_function_shape": self.decision_function_shape,
            }
        )
        return params

    def _attrs_from_cpu(self, model):
        n_classes = len(model.classes_)
        if n_classes > 2:
            raise UnsupportedOnGPU("multiclass models are not supported")
        return {
            "n_classes_": n_classes,
            "classes_": model.classes_,
            "class_weight_": model.class_weight_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        if hasattr(self, "_multiclass"):
            raise UnsupportedOnCPU(
                "Converting multiclass models from GPU is not yet supported"
            )

        out = super()._attrs_to_cpu(model)

        # sklearn's binary classification expects inverted
        # _dual_coef_ and _intercept_.
        out.update(
            _dual_coef_=-out["dual_coef_"],
            _intercept_=-out["intercept_"],
            classes_=self.classes_,
            class_weight_=self.class_weight_,
        )
        return out

    def __init__(
        self,
        *,
        handle=None,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        cache_size=1024.0,
        max_iter=-1,
        nochange_steps=1000,
        verbose=False,
        output_type=None,
        probability=False,
        random_state=None,
        class_weight=None,
        decision_function_shape="ovo",
    ):
        super().__init__(
            handle=handle,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            cache_size=cache_size,
            max_iter=max_iter,
            nochange_steps=nochange_steps,
            verbose=verbose,
            output_type=output_type,
        )
        self.probability = probability
        self.random_state = random_state
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape

    @property
    @reflect
    def support_(self):
        if hasattr(self, "_multiclass"):
            estimators = self._multiclass.multiclass_estimator.estimators_
            return cp.concatenate(
                [cp.asarray(cls._support_) for cls in estimators]
            )
        else:
            return self._support_

    @support_.setter
    def support_(self, value):
        self._support_ = value

    @property
    @reflect
    def intercept_(self):
        if hasattr(self, "_multiclass"):
            estimators = self._multiclass.multiclass_estimator.estimators_
            return cp.concatenate(
                [cp.asarray(cls._intercept_) for cls in estimators]
            )
        else:
            return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value

    def _fit_multiclass(self, X, y, sample_weight):
        if sample_weight is not None:
            warn(
                "Sample weights are currently ignored for multi class classification"
            )

        params = self.get_params()
        decision_function_shape = params.pop("decision_function_shape")
        wrappers = {"ovo": OneVsOneClassifier, "ovr": OneVsRestClassifier}
        if (multiclass_cls := wrappers.get(decision_function_shape)) is None:
            raise ValueError(
                f"Expected `decision_function_shape` to be one of "
                f"{list(wrappers)}, got {decision_function_shape}"
            )
        self._multiclass = multiclass_cls(
            estimator=SVC(**params),
            handle=self.handle,
            verbose=self.verbose,
            output_type=self.output_type,
        )
        self._multiclass.fit(X, y)

        # if using one-vs-one we align support_ indices to those of
        # full dataset
        if decision_function_shape == "ovo":
            y = cp.array(y)
            classes = cp.unique(y)
            n_classes = len(classes)
            estimator_index = 0
            # Loop through multiclass estimators and re-align support_ indices
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    cond = cp.logical_or(y == classes[i], y == classes[j])
                    ovo_support = cp.array(
                        self._multiclass.multiclass_estimator.estimators_[
                            estimator_index
                        ].support_
                    )
                    self._multiclass.multiclass_estimator.estimators_[
                        estimator_index
                    ].support_ = cp.nonzero(cond)[0][ovo_support]
                    estimator_index += 1

        self.shape_fit_ = X.shape
        self.fit_status_ = 0
        self.n_iter_ = np.concatenate(
            [
                est.n_iter_
                for est in self._multiclass.multiclass_estimator.estimators_
            ]
        )
        return self

    def _fit_proba(self, X, y, sample_weight):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold

        params = {
            **self.get_params(),
            "probability": False,
            "output_type": "numpy",
            "class_weight": None,
        }

        # Currently CalibratedClassifierCV expects data on the host, see
        # https://github.com/rapidsai/cuml/issues/2608
        X = input_to_host_array_with_sparse_support(X)

        if sample_weight is not None:
            sample_weight = sample_weight.to_output("numpy")

        y = input_to_host_array(y).array

        cv = StratifiedKFold(
            n_splits=5,
            random_state=check_random_seed(self.random_state),
            shuffle=True,
        )
        cccv = CalibratedClassifierCV(SVC(**params), cv=cv, ensemble=False)

        with exit_internal_context():
            cccv.fit(X, y, sample_weight=sample_weight)

        cal_clf = cccv.calibrated_classifiers_[0]
        svc = cal_clf.estimator

        self._probA = np.array([cal.a_ for cal in cal_clf.calibrators])
        self._probB = np.array([cal.b_ for cal in cal_clf.calibrators])

        if hasattr(svc, "_multiclass"):
            attrs = ["_multiclass", "fit_status_", "shape_fit_"]
        else:
            attrs = [
                "support_",
                "support_vectors_",
                "dual_coef_",
                "intercept_",
                "n_support_",
                "fit_status_",
                "shape_fit_",
                "n_iter_",
                "_gamma",
                "_sparse",
            ]

        # Forward on inner attributes
        for attr in attrs:
            setattr(self, attr, getattr(svc, attr))

        return self

    @generate_docstring(y="dense_anydtype")
    @reflect(reset=True)
    def fit(self, X, y, sample_weight=None, *, convert_dtype=True) -> "SVC":
        """
        Fit the model with X and y.

        """
        if hasattr(self, "_multiclass"):
            del self._multiclass

        y, classes = preprocess_labels(y)
        if len(classes) == 1:
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, but "
                "the data contains only 1 class"
            )
        self.n_classes_ = len(classes)
        self.classes_ = classes
        self.class_weight_, sample_weight = process_class_weight(
            classes,
            y,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
            float64=(getattr(X, "dtype", np.float32) == np.float64),
            balanced_with_sample_weight=False,
        )

        if self.probability:
            return self._fit_proba(X, y, sample_weight)

        if len(classes) > 2:
            return self._fit_multiclass(X, y, sample_weight)

        if is_sparse(X):
            X = SparseCumlArray(
                X,
                convert_to_dtype=(
                    None if X.dtype in (np.float32, np.float64) else np.float32
                ),
                check_rows=y.shape[0],
            )
        else:
            X = input_to_cuml_array(
                X,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                check_dtype=[np.float32, np.float64],
                check_rows=y.shape[0],
                order="F",
            ).array

        # Encode y to -1/1 (like [0, 1, 0, 1] -> [-1, 1, -1, 1])
        y = CumlArray(data=cp.array([-1, 1], dtype=X.dtype).take(y))

        self._fit(X, y, sample_weight)

        return self

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @run_in_internal_context
    def predict(self, X, *, convert_dtype=True):
        """
        Predicts the class labels for X. The returned y values are the class
        labels associated to sign(decision_function(X)).
        """
        if hasattr(self, "_multiclass"):
            inds = self._multiclass.predict(X).to_output("cupy")
        elif self.probability:
            probs = self.predict_proba(X).to_output("cupy")
            inds = cp.argmax(probs, axis=1)
        else:
            res = self.decision_function(X, convert_dtype=convert_dtype)
            inds = (res.to_output("cupy") >= 0).view(cp.int8)

        with exit_internal_context():
            output_type = self._get_output_type(X)
        return decode_labels(inds, self.classes_, output_type=output_type)

    @generate_docstring(
        skip_parameters_heading=True,
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted probabilities",
            "shape": "(n_samples, n_classes)",
        },
    )
    @reflect
    def predict_proba(self, X, *, log=False) -> CumlArray:
        """
        Predicts the class probabilities for X.

        The model has to be trained with probability=True to use this method.

        Parameters
        ----------
        log: boolean (default = False)
             Whether to return log probabilities.

        """
        from cupyx.scipy.special import expit

        if not self.probability:
            raise NotFittedError(
                "This classifier is not fitted to predict "
                "probabilities. Fit a new classifier with "
                "probability=True to enable predict_proba."
            )
        preds = self.decision_function(X).to_output("cupy")
        if preds.ndim == 1:
            preds = preds[:, None]

        n_classes = len(self.classes_)

        proba = cp.zeros((preds.shape[0], n_classes))
        for i in range(preds.shape[1]):
            a = self._probA[i]
            b = self._probB[i]
            ind = i + 1 if n_classes == 2 else i
            proba[:, ind] = expit(-(a * preds[:, i] + b))

        if n_classes == 2:
            proba[:, 0] = 1.0 - proba[:, 1]
        else:
            den = cp.sum(proba, axis=1)
            cp.divide(proba, den[:, None], out=proba)
            # If all probabilities are 0, use a uniform distribution
            proba[den == 0] = 1 / n_classes
            # Clip to between 0 and 1 to handle rounding error
            cp.clip(proba, 0, 1, out=proba)

        if log:
            proba = cp.log(proba)

        return CumlArray(data=proba)

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Log of predicted probabilities",
            "shape": "(n_samples, n_classes)",
        }
    )
    @reflect
    def predict_log_proba(self, X) -> CumlArray:
        """
        Predicts the log probabilities for X (returns log(predict_proba(x)).

        The model has to be trained with probability=True to use this method.

        """
        return self.predict_proba(X, log=True)

    @generate_docstring(
        return_values={
            "name": "results",
            "type": "dense",
            "description": "Decision function values",
            "shape": "(n_samples, 1)",
        }
    )
    @reflect
    def decision_function(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Calculates the decision function values for X.

        """
        if hasattr(self, "_multiclass"):
            return self._multiclass.decision_function(X)

        dtype = self.support_vectors_.dtype

        if is_sparse(X):
            X = SparseCumlArray(X, convert_to_dtype=dtype)
        else:
            X = input_to_cuml_array(
                X,
                check_dtype=[dtype],
                convert_to_dtype=(dtype if convert_dtype else None),
                order="F",
                check_cols=self.support_vectors_.shape[1],
            ).array

        return self._predict(X)
